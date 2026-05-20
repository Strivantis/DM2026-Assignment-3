"""
dataset.py
----------
Data loading, preprocessing, normalization, and augmentation pipeline
for the HAR 1D-ResNet project.

Data format:
  - Each CSV: 300 rows × columns [index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id]
  - Raw feature shape after loading:  (300, 6)
  - Pruned + magnitude channel:       (300, 4)  →  transposed to (4, 300) for 1D-CNN
  - Train users: User_001 – User_060
  - Test  users: User_061 – User_100

Feature Engineering (v6 — Radical Pruning)
--------------------------------------------
  Multicollinear std_x / std_y / std_z axes are dropped entirely to resolve
  extreme weight-thrashing in the 1D-CNN layers.  The synthesised mean_mag
  feature is also dropped (low importance, redundant with raw axes).  Only the
  four highest-signal channels are retained:

   Pruned Channel Map (4 channels):
     [0] mean_x   – raw X-axis sliding-window mean
     [1] mean_y   – raw Y-axis sliding-window mean
     [2] mean_z   – raw Z-axis sliding-window mean
     [3] std_mag  = sqrt(std_x² + std_y² + std_z² + 1e-8)
                    holistic dynamic-motion energy (rotation-invariant)

  Final output shape per sample: (4, 300).

Instance-Level Mean Centering for Pose Elimination (v6)
---------------------------------------------------------
  To eradicate device-wearing orientation bias (user-to-user covariate shift),
  per-sample temporal mean centering is applied to the three directional axes
  (channels 0–2) BEFORE global Z-score normalization:

      X_ch ← X_ch - mean(X_ch)    for ch ∈ {mean_x, mean_y, mean_z}

  std_mag (channel 3) is deliberately excluded — its absolute vibration
  magnitude must be preserved for correct dynamic-motion representation.

  Execution order:
    1. Load raw (N, 300, 6) array
    2. Compute std_mag, prune to (N, 300, 4)
    3. Apply instance mean centering to channels 0–2   ← NEW (v6)
    4. Compute per-fold Z-score statistics (training fold only)
    5. Apply global Z-score normalisation

Augmentation (v3/v5, unchanged)
---------------------------------
  Applied to ALL training samples (global, not minority-only) with probability
  AUG_PROB per technique to prevent subject-signature memorisation.

  PROTECTED EXEMPTION (v3): Label 2 samples are STRICTLY PROHIBITED from
  receiving Time Masking (Cutout 1D) to preserve critical low-frequency
  structural signals that distinguish Label 2 from Label 1.

  • Time Masking (Cutout 1D)  – zero out a random window of 15-30 time-steps
                                 [DISABLED for label == 2]
  • Channel Masking (Sensor Dropout) – zero out 1–2 random feature channels
  • Jitter                    – add Gaussian noise
  • Scale                     – multiply by per-channel random scalar

  All augmentations are disabled during validation and testing.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
FEATURE_COLS  = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
RAW_CHANNELS  = len(FEATURE_COLS)   # 6  (original sensor columns in CSV)
N_CHANNELS    = 4                   # v6: 4 pruned channels (mean_x/y/z + std_mag)
SEQ_LEN       = 300
N_CLASSES     = 6
N_FOLDS       = 5

# ── Augmentation hyper-parameters ─────────────────────────────────────────────
# Global execution probability: each augmentation technique fires independently
AUG_PROB            = 0.5

# Time masking (Cutout 1D)
TIME_MASK_MIN       = 15   # minimum masked window length (time steps)
TIME_MASK_MAX       = 30   # maximum masked window length (time steps)

# Channel masking (Sensor Dropout)
CHANNEL_MASK_MIN    = 1    # minimum number of channels to zero out
CHANNEL_MASK_MAX    = 2    # maximum number of channels to zero out

# Jitter noise std and scale range
JITTER_SIGMA = 0.05
SCALE_LOW    = 0.9
SCALE_HIGH   = 1.1

# Protected label: Time Masking is strictly disabled for this class
PROTECTED_LABEL = 2


# ──────────────────────────────────────────
# Pruned feature engineering  (v6)
# ──────────────────────────────────────────

def build_pruned_channels(sig: np.ndarray) -> np.ndarray:
    """
    Construct the 4-channel pruned feature matrix from the raw 6-channel input.

    Drops std_x, std_y, std_z (multicollinear, correlation 0.93–0.98) and
    mean_mag (low importance).  Retains mean_x/y/z (raw directional signal)
    plus std_mag (rotation-invariant motion-energy scalar).

    Parameters
    ----------
    sig : np.ndarray, shape (300, 6)
          Columns: [mean_x, mean_y, mean_z, std_x, std_y, std_z]

    Returns
    -------
    out : np.ndarray, shape (300, 4)
          Columns: [mean_x, mean_y, mean_z, std_mag]
    """
    mean_x, mean_y, mean_z = sig[:, 0], sig[:, 1], sig[:, 2]
    std_x,  std_y,  std_z  = sig[:, 3], sig[:, 4], sig[:, 5]

    std_mag = np.sqrt(std_x ** 2 + std_y ** 2 + std_z ** 2 + 1e-8).astype(np.float32)
    std_mag = std_mag[:, np.newaxis]   # (300, 1)

    # Stack: mean_x / mean_y / mean_z already in sig[:, 0:3]
    out = np.concatenate([sig[:, 0:3], std_mag], axis=1)   # (300, 4)
    return out


# ──────────────────────────────────────────
# Instance-level mean centering  (v6)
# ──────────────────────────────────────────

def apply_instance_centering(signals: np.ndarray) -> np.ndarray:
    """
    Apply sample-wise temporal mean centering to the three directional axes
    (channels 0–2: mean_x, mean_y, mean_z) to eradicate device-placement
    orientation bias across users.

    For every individual 300-timestep sequence X:
        X[:, ch] ← X[:, ch] - mean(X[:, ch])    for ch in {0, 1, 2}

    Channel 3 (std_mag) is INTENTIONALLY EXCLUDED — its absolute vibration
    magnitude must be preserved for correct dynamic-motion representation.

    This centering is executed BEFORE global Z-score normalization so that
    fold-level statistics are computed on orientation-corrected signals.

    Parameters
    ----------
    signals : np.ndarray, shape (N, 300, 4)
              Raw pruned feature arrays (not yet normalised).

    Returns
    -------
    centered : np.ndarray, shape (N, 300, 4)
               Channels 0–2 centered per-sample; channel 3 unchanged.
    """
    centered = signals.copy()
    # Center channels 0, 1, 2 independently for each sample
    # mean over 300 timesteps: axis=1 → shape (N, 1) after keepdims
    centered[:, :, 0:3] -= centered[:, :, 0:3].mean(axis=1, keepdims=True)
    return centered


# ──────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────

def load_all_samples(root_dir: str):
    """
    Walk *root_dir* and load every CSV file.

    Returns
    -------
    signals  : np.ndarray, shape (N, 300, 4)  – pruned 4-channel feature array
                                                 (NOT yet instance-centred or normalised)
    labels   : np.ndarray, shape (N,)          – integer class index
    groups   : np.ndarray, shape (N,)          – integer user id (for GroupKFold)
    file_ids : np.ndarray, shape (N,)          – original file_id value (for submission)
    """
    csv_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found under: {root_dir}")

    signals_list  = []
    labels_list   = []
    groups_list   = []
    file_ids_list = []

    for path in csv_files:
        df = pd.read_csv(path)

        # --- raw feature matrix (300, 6)
        sig_raw = df[FEATURE_COLS].values.astype(np.float32)
        assert sig_raw.shape == (SEQ_LEN, RAW_CHANNELS), \
            f"Unexpected shape {sig_raw.shape} for file {path}"

        # --- prune to 4-channel representation: [mean_x, mean_y, mean_z, std_mag]
        sig = build_pruned_channels(sig_raw)          # (300, 4)
        assert sig.shape == (SEQ_LEN, N_CHANNELS), \
            f"Unexpected post-pruning shape {sig.shape} for file {path}"

        # --- label (one label per file, guaranteed consistent)
        lbl = int(df["label"].iloc[0])

        # --- group: derive integer user id from directory name
        # e.g. ".../User_042/00017.csv"  →  42
        user_dir  = os.path.basename(os.path.dirname(path))   # "User_042"
        user_id   = int(user_dir.split("_")[1])                # 42

        # --- file_id for submission
        file_id = int(df["file_id"].iloc[0])

        signals_list.append(sig)
        labels_list.append(lbl)
        groups_list.append(user_id)
        file_ids_list.append(file_id)

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 4)
    labels   = np.array(labels_list,   dtype=np.int64)
    groups   = np.array(groups_list,   dtype=np.int64)
    file_ids = np.array(file_ids_list, dtype=np.int64)

    return signals, labels, groups, file_ids


def compute_normalization_params(signals: np.ndarray):
    """
    Compute per-channel mean and std from a set of signals.

    Parameters
    ----------
    signals : np.ndarray, shape (N, 300, 4)

    Returns
    -------
    mean : np.ndarray, shape (4,)
    std  : np.ndarray, shape (4,)
    """
    # Reshape to (N*300, 4) then compute stats along axis 0
    flat = signals.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)   # avoid division by zero
    return mean, std


def normalize(signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization using *pre-computed* mean/std.
    signals : (N, 300, 4)  →  returns (N, 300, 4)
    """
    return (signals - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]


def get_fold_splits(signals, labels, groups, fold_idx: int = 0):
    """
    Split data into (train_idx, val_idx) for the requested fold using
    StratifiedGroupKFold, ensuring the same user never spans both splits.

    Returns
    -------
    train_idx, val_idx : 1-D integer arrays
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(sgkf.split(signals, labels, groups=groups))
    train_idx, val_idx = splits[fold_idx]
    return train_idx, val_idx


# ──────────────────────────────────────────
# Augmentation helpers (applied per sample)
# ──────────────────────────────────────────

def augment_jitter(signal: np.ndarray) -> np.ndarray:
    """Add Gaussian noise to simulate sensor variance. signal: (300, 4)"""
    noise = np.random.normal(0.0, JITTER_SIGMA, size=signal.shape).astype(np.float32)
    return signal + noise


def augment_scale(signal: np.ndarray) -> np.ndarray:
    """Multiply by random per-channel scalar in [SCALE_LOW, SCALE_HIGH]. signal: (300, 4)"""
    scale = np.random.uniform(SCALE_LOW, SCALE_HIGH, size=(1, N_CHANNELS)).astype(np.float32)
    return signal * scale


def augment_time_masking(signal: np.ndarray) -> np.ndarray:
    """
    Cutout 1D – zero out a contiguous window of TIME_MASK_MIN to TIME_MASK_MAX
    time steps chosen uniformly at random.

    Parameters
    ----------
    signal : np.ndarray, shape (300, 4)

    Returns
    -------
    augmented signal : np.ndarray, shape (300, 4)
    """
    signal = signal.copy()
    mask_len   = np.random.randint(TIME_MASK_MIN, TIME_MASK_MAX + 1)
    start      = np.random.randint(0, SEQ_LEN - mask_len + 1)
    signal[start : start + mask_len, :] = 0.0
    return signal


def augment_channel_masking(signal: np.ndarray) -> np.ndarray:
    """
    Sensor Dropout – zero out CHANNEL_MASK_MIN to CHANNEL_MASK_MAX feature
    channels chosen uniformly at random (without replacement).

    Parameters
    ----------
    signal : np.ndarray, shape (300, 4)

    Returns
    -------
    augmented signal : np.ndarray, shape (300, 4)
    """
    signal       = signal.copy()
    n_mask       = np.random.randint(CHANNEL_MASK_MIN, CHANNEL_MASK_MAX + 1)
    channels     = np.random.choice(N_CHANNELS, size=n_mask, replace=False)
    signal[:, channels] = 0.0
    return signal


def apply_augmentation(signal: np.ndarray, label: int) -> np.ndarray:
    """
    Apply all augmentations independently with probability AUG_PROB each.
    Operates on signal of shape (300, 4) and returns the same shape.

    Protected Exemption (v3/v5/v6):
      If label == PROTECTED_LABEL (2), Time Masking (Cutout 1D) is
      strictly skipped to preserve critical low-frequency structural signals
      that differentiate Label 2 from Label 1.

    Augmentation order:
      1. Time Masking  (Cutout 1D)    ← SKIPPED if label == 2
      2. Channel Masking (Sensor Dropout)
      3. Jitter
      4. Scale
    """
    # Time Masking: disabled for PROTECTED_LABEL to avoid structural signal loss
    if label != PROTECTED_LABEL:
        if np.random.rand() < AUG_PROB:
            signal = augment_time_masking(signal)

    if np.random.rand() < AUG_PROB:
        signal = augment_channel_masking(signal)
    if np.random.rand() < AUG_PROB:
        signal = augment_jitter(signal)
    if np.random.rand() < AUG_PROB:
        signal = augment_scale(signal)
    return signal


# ──────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────

class HARDataset(Dataset):
    """
    PyTorch Dataset for HAR sequences.

    Parameters
    ----------
    signals    : np.ndarray, shape (N, 300, 4)  – already instance-centred and normalised
    labels     : np.ndarray, shape (N,)
    is_train   : bool  – enables global augmentation for ALL classes during training
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, is_train: bool = False):
        self.signals  = signals.astype(np.float32)  # (N, 300, 4)
        self.labels   = labels.astype(np.int64)
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()   # (300, 4)
        label  = self.labels[idx]

        # Apply augmentation during training for ALL classes (global augmentation).
        # Individual techniques fire stochastically at probability AUG_PROB each,
        # preventing the model from memorising majority-class subject signatures.
        # EXCEPTION: Time Masking is strictly skipped when label == PROTECTED_LABEL (2).
        if self.is_train:
            signal = apply_augmentation(signal, int(label))

        # Transpose: (300, 4)  →  (4, 300)  for 1D-CNN input (C, L)
        signal = signal.T  # (4, 300)

        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


class HARTestDataset(Dataset):
    """
    Dataset for the held-out test set (no labels, no augmentation).

    Parameters
    ----------
    signals  : np.ndarray, shape (N, 300, 4)  – already instance-centred and normalised
    file_ids : np.ndarray, shape (N,)           – for submission mapping
    """

    def __init__(self, signals: np.ndarray, file_ids: np.ndarray):
        self.signals  = signals.astype(np.float32)
        self.file_ids = file_ids

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].T.copy()   # (4, 300)
        return torch.from_numpy(signal), self.file_ids[idx]


# ──────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────

def build_fold_datasets(train_root: str, fold_idx: int = 0):
    """
    Full pipeline for one CV fold.

    Processing order (v6):
      1. Load raw feature arrays → (N, 300, 4)  [mean_x, mean_y, mean_z, std_mag]
      2. Apply instance mean centering to channels 0–2 per sample (BEFORE global stats)
      3. Compute per-fold Z-score statistics from training fold only  (no leakage)
      4. Apply global Z-score normalisation to train and validation splits

    Returns
    -------
    train_dataset : HARDataset  (global augmentation ON, with L2 time-mask protection)
    val_dataset   : HARDataset  (no augmentation)
    norm_params   : (mean, std) tuple – derived from training fold only
    class_counts  : np.ndarray, shape (N_CLASSES,) – for loss weighting
    """
    signals, labels, groups, _ = load_all_samples(train_root)

    train_idx, val_idx = get_fold_splits(signals, labels, groups, fold_idx)

    # ── v6: Instance mean centering BEFORE global Z-score statistics ──────────
    # Apply per-sample centering on the full signal array (both splits).
    # This ensures fold-level normalization statistics are computed on
    # orientation-corrected data.
    signals = apply_instance_centering(signals)

    # Compute normalisation from training fold only → no leakage
    mean, std = compute_normalization_params(signals[train_idx])

    train_signals = normalize(signals[train_idx], mean, std)
    val_signals   = normalize(signals[val_idx],   mean, std)

    train_labels  = labels[train_idx]
    val_labels    = labels[val_idx]

    # Class counts for loss weighting / logging
    class_counts = np.bincount(train_labels, minlength=N_CLASSES)

    train_dataset = HARDataset(train_signals, train_labels, is_train=True)
    val_dataset   = HARDataset(val_signals,   val_labels,   is_train=False)

    return train_dataset, val_dataset, (mean, std), class_counts


def load_test_samples(root_dir: str):
    """
    Walk *root_dir* and load every CSV file for the **test** set.
    Test CSVs have NO 'label' column: index, mean_x, …, std_z, file_id.

    Returns
    -------
    signals  : np.ndarray, shape (N, 300, 4)  – pruned 4-channel array
                                                 (NOT yet instance-centred or normalised)
    file_ids : np.ndarray, shape (N,)
    """
    csv_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found under: {root_dir}")

    signals_list  = []
    file_ids_list = []

    for path in csv_files:
        df = pd.read_csv(path)

        sig_raw = df[FEATURE_COLS].values.astype(np.float32)  # (300, 6)
        assert sig_raw.shape == (SEQ_LEN, RAW_CHANNELS), \
            f"Unexpected shape {sig_raw.shape} for file {path}"

        # Prune to 4-channel representation
        sig = build_pruned_channels(sig_raw)   # (300, 4)

        file_id = int(df["file_id"].iloc[0])

        signals_list.append(sig)
        file_ids_list.append(file_id)

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 4)
    file_ids = np.array(file_ids_list, dtype=np.int64)

    return signals, file_ids


def build_test_dataset(test_root: str, norm_mean: np.ndarray, norm_std: np.ndarray):
    """
    Build the test dataset using normalisation parameters from the training fold.

    Processing order (v6):
      1. Load raw test signals → (N, 300, 4)
      2. Apply instance mean centering to channels 0–2 (mirrors training pipeline)
      3. Apply global Z-score normalisation using fold-derived mean/std

    Returns
    -------
    test_dataset : HARTestDataset
    file_ids     : np.ndarray
    """
    signals, file_ids = load_test_samples(test_root)

    # ── v6: Instance mean centering BEFORE global normalisation ──────────────
    signals = apply_instance_centering(signals)

    test_signals = normalize(signals, norm_mean, norm_std)
    return HARTestDataset(test_signals, file_ids), file_ids
