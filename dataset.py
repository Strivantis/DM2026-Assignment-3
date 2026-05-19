"""
dataset.py
----------
Data loading, preprocessing, normalization, and augmentation pipeline
for the HAR 1D-ResNet project.

Data format:
  - Each CSV: 300 rows × columns [index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id]
  - Raw feature shape after loading: (300, 6)
  - Magnitude features appended: (300, 8)  →  transposed to (8, 300) for 1D-CNN
  - Train users: User_001 – User_060
  - Test  users: User_061 – User_100

Feature Engineering (v3)
-------------------------
  Two rotation-invariant magnitude channels are appended after the raw 6-channel
  features to decouple activity identification from subject device orientation:

  • mean_mag = sqrt(mean_x² + mean_y² + mean_z² + 1e-8)
  • std_mag  = sqrt(std_x²  + std_y²  + std_z²  + 1e-8)

  Final channel layout (8 channels):
    [0] mean_x   [1] mean_y   [2] mean_z
    [3] std_x    [4] std_y    [5] std_z
    [6] mean_mag [7] std_mag

Augmentation (v3)
-----------------
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
FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
RAW_CHANNELS  = len(FEATURE_COLS)   # 6  (original sensor channels)
N_CHANNELS    = 8                   # 6 raw + 2 magnitude channels
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
# Magnitude feature engineering
# ──────────────────────────────────────────

def append_magnitude_channels(sig: np.ndarray) -> np.ndarray:
    """
    Append two rotation-invariant magnitude channels to the raw feature matrix.

    Parameters
    ----------
    sig : np.ndarray, shape (300, 6)
          Columns: [mean_x, mean_y, mean_z, std_x, std_y, std_z]

    Returns
    -------
    sig_aug : np.ndarray, shape (300, 8)
              Columns: [mean_x, mean_y, mean_z, std_x, std_y, std_z,
                        mean_mag, std_mag]
    """
    mean_x, mean_y, mean_z = sig[:, 0], sig[:, 1], sig[:, 2]
    std_x,  std_y,  std_z  = sig[:, 3], sig[:, 4], sig[:, 5]

    mean_mag = np.sqrt(mean_x ** 2 + mean_y ** 2 + mean_z ** 2 + 1e-8).astype(np.float32)
    std_mag  = np.sqrt(std_x  ** 2 + std_y  ** 2 + std_z  ** 2 + 1e-8).astype(np.float32)

    # Expand dims for concatenation: (300,) → (300, 1)
    mean_mag = mean_mag[:, np.newaxis]
    std_mag  = std_mag[:, np.newaxis]

    return np.concatenate([sig, mean_mag, std_mag], axis=1)   # (300, 8)


# ──────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────

def load_all_samples(root_dir: str):
    """
    Walk *root_dir* and load every CSV file.

    Returns
    -------
    signals : np.ndarray, shape (N, 300, 8)   – 6 raw + 2 magnitude channels
    labels  : np.ndarray, shape (N,)   – integer class index
    groups  : np.ndarray, shape (N,)   – integer user id (for GroupKFold)
    file_ids: np.ndarray, shape (N,)   – original file_id value (for submission)
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

        # --- append magnitude channels → (300, 8)
        sig = append_magnitude_channels(sig_raw)
        assert sig.shape == (SEQ_LEN, N_CHANNELS), \
            f"Unexpected post-magnitude shape {sig.shape} for file {path}"

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

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 8)
    labels   = np.array(labels_list,   dtype=np.int64)
    groups   = np.array(groups_list,   dtype=np.int64)
    file_ids = np.array(file_ids_list, dtype=np.int64)

    return signals, labels, groups, file_ids


def compute_normalization_params(signals: np.ndarray):
    """
    Compute per-channel mean and std from a set of signals.

    Parameters
    ----------
    signals : np.ndarray, shape (N, 300, 8)

    Returns
    -------
    mean : np.ndarray, shape (8,)
    std  : np.ndarray, shape (8,)
    """
    # Reshape to (N*300, 8) then compute stats along axis 0
    flat = signals.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)   # avoid division by zero
    return mean, std


def normalize(signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization using *pre-computed* mean/std.
    signals : (N, 300, 8)  →  returns (N, 300, 8)
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
    """Add Gaussian noise to simulate sensor variance. signal: (300, 8)"""
    noise = np.random.normal(0.0, JITTER_SIGMA, size=signal.shape).astype(np.float32)
    return signal + noise


def augment_scale(signal: np.ndarray) -> np.ndarray:
    """Multiply by random per-channel scalar in [SCALE_LOW, SCALE_HIGH]. signal: (300, 8)"""
    scale = np.random.uniform(SCALE_LOW, SCALE_HIGH, size=(1, N_CHANNELS)).astype(np.float32)
    return signal * scale


def augment_time_masking(signal: np.ndarray) -> np.ndarray:
    """
    Cutout 1D – zero out a contiguous window of TIME_MASK_MIN to TIME_MASK_MAX
    time steps chosen uniformly at random.

    Parameters
    ----------
    signal : np.ndarray, shape (300, 8)

    Returns
    -------
    augmented signal : np.ndarray, shape (300, 8)
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
    signal : np.ndarray, shape (300, 8)

    Returns
    -------
    augmented signal : np.ndarray, shape (300, 8)
    """
    signal       = signal.copy()
    n_mask       = np.random.randint(CHANNEL_MASK_MIN, CHANNEL_MASK_MAX + 1)
    channels     = np.random.choice(N_CHANNELS, size=n_mask, replace=False)
    signal[:, channels] = 0.0
    return signal


def apply_augmentation(signal: np.ndarray, label: int) -> np.ndarray:
    """
    Apply all augmentations independently with probability AUG_PROB each.
    Operates on signal of shape (300, 8) and returns the same shape.

    Protected Exemption (v3):
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
    signals    : np.ndarray, shape (N, 300, 8)  – already normalised
    labels     : np.ndarray, shape (N,)
    is_train   : bool  – enables global augmentation for ALL classes during training
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, is_train: bool = False):
        self.signals  = signals.astype(np.float32)  # (N, 300, 8)
        self.labels   = labels.astype(np.int64)
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()   # (300, 8)
        label  = self.labels[idx]

        # Apply augmentation during training for ALL classes (global augmentation).
        # Individual techniques fire stochastically at probability AUG_PROB each,
        # preventing the model from memorising majority-class subject signatures.
        # EXCEPTION: Time Masking is strictly skipped when label == PROTECTED_LABEL (2).
        if self.is_train:
            signal = apply_augmentation(signal, int(label))

        # Transpose: (300, 8)  →  (8, 300)  for 1D-CNN input (C, L)
        signal = signal.T  # (8, 300)

        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


class HARTestDataset(Dataset):
    """
    Dataset for the held-out test set (no labels, no augmentation).

    Parameters
    ----------
    signals  : np.ndarray, shape (N, 300, 8)  – already normalised
    file_ids : np.ndarray, shape (N,)          – for submission mapping
    """

    def __init__(self, signals: np.ndarray, file_ids: np.ndarray):
        self.signals  = signals.astype(np.float32)
        self.file_ids = file_ids

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].T.copy()   # (8, 300)
        return torch.from_numpy(signal), self.file_ids[idx]


# ──────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────

def build_fold_datasets(train_root: str, fold_idx: int = 0):
    """
    Full pipeline for one CV fold.

    Returns
    -------
    train_dataset : HARDataset  (global augmentation ON, with L2 time-mask protection)
    val_dataset   : HARDataset  (no augmentation)
    norm_params   : (mean, std) tuple – derived from training fold only
    class_counts  : np.ndarray, shape (N_CLASSES,) – for loss weighting
    """
    signals, labels, groups, _ = load_all_samples(train_root)

    train_idx, val_idx = get_fold_splits(signals, labels, groups, fold_idx)

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
    signals  : np.ndarray, shape (N, 300, 8)   – 6 raw + 2 magnitude channels
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

        # Append magnitude channels → (300, 8)
        sig = append_magnitude_channels(sig_raw)

        file_id = int(df["file_id"].iloc[0])

        signals_list.append(sig)
        file_ids_list.append(file_id)

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 8)
    file_ids = np.array(file_ids_list, dtype=np.int64)

    return signals, file_ids


def build_test_dataset(test_root: str, norm_mean: np.ndarray, norm_std: np.ndarray):
    """
    Build the test dataset using normalisation parameters from the training fold.

    Returns
    -------
    test_dataset : HARTestDataset
    file_ids     : np.ndarray
    """
    signals, file_ids = load_test_samples(test_root)
    test_signals = normalize(signals, norm_mean, norm_std)
    return HARTestDataset(test_signals, file_ids), file_ids
