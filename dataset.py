"""
dataset.py
----------
Data loading, preprocessing, normalization, and augmentation pipeline
for the HAR 1D-ResNet project.

Data format:
  - Each CSV: 300 rows × columns [index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id]
  - Raw feature shape after loading:  (300, 6)
  - 8-channel feature matrix:         (300, 8)  →  transposed to (8, 300) for 1D-CNN

Channel map (8 channels):
  [0] mean_x, [1] mean_y, [2] mean_z  – raw sliding-window means
  [3] std_x,  [4] std_y,  [5] std_z   – raw sliding-window std deviations
  [6] mean_mag = sqrt(mean_x² + mean_y² + mean_z² + 1e-8)  (direction magnitude)
  [7] std_mag  = sqrt(std_x²  + std_y²  + std_z²  + 1e-8)  (motion energy)

Label modes (build_fold_datasets `mode` argument):
  'stage1' – 5-class: L0→0, L1/L2→1, L3→2, L4→3, L5→4
  'stage2' – 2-class specialist: L1→0, L2→1 (other samples filtered out)
  'flat'   – original 6-class labels unchanged

Augmentation: applied to all training samples with probability AUG_PROB per
technique. Label 2 is exempt from Time Masking (checked against original label
before any remapping). All augmentations disabled during validation/testing.
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
N_CHANNELS    = 8                   # v7: 8 stable channels (mean_x/y/z + std_x/y/z + mean_mag + std_mag)
SEQ_LEN       = 300
N_CLASSES     = 6                   # original label space (0–5)
N_CLASSES_S1  = 5                   # Stage 1: 5-class generalist (L1+L2 merged)
N_CLASSES_S2  = 2                   # Stage 2: 2-class specialist (L1 vs L2)
N_FOLDS       = 5

# Stage 1 label mapping: original → stage1 index
# L0→0, L1→1, L2→1, L3→2, L4→3, L5→4
STAGE1_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}

# Stage 2 label mapping: original L1/L2 only → binary index
# L1→0, L2→1
STAGE2_MAP = {1: 0, 2: 1}

# ── Augmentation hyper-parameters ─────────────────────────────────────────────
AUG_PROB         = 0.5
TIME_MASK_MIN    = 15
TIME_MASK_MAX    = 30
CHANNEL_MASK_MIN = 1
CHANNEL_MASK_MAX = 2
JITTER_SIGMA     = 0.05
SCALE_LOW        = 0.9
SCALE_HIGH       = 1.1

# Original label whose time masking is protected (before any remapping)
PROTECTED_ORIG_LABEL = 2


# ──────────────────────────────────────────
# 8-Channel feature engineering  (v7 stable baseline)
# ──────────────────────────────────────────

def build_feature_channels(sig: np.ndarray) -> np.ndarray:
    """
    Construct the 8-channel feature matrix from the raw 6-channel input.

    Adds:
      [6] mean_mag = sqrt(mean_x² + mean_y² + mean_z² + 1e-8)
      [7] std_mag  = sqrt(std_x²  + std_y²  + std_z²  + 1e-8)

    Parameters
    ----------
    sig : np.ndarray, shape (300, 6)
          Columns: [mean_x, mean_y, mean_z, std_x, std_y, std_z]

    Returns
    -------
    out : np.ndarray, shape (300, 8)
          Columns: [mean_x, mean_y, mean_z, std_x, std_y, std_z, mean_mag, std_mag]
    """
    mean_x, mean_y, mean_z = sig[:, 0], sig[:, 1], sig[:, 2]
    std_x,  std_y,  std_z  = sig[:, 3], sig[:, 4], sig[:, 5]

    mean_mag = np.sqrt(mean_x ** 2 + mean_y ** 2 + mean_z ** 2 + 1e-8).astype(np.float32)
    std_mag  = np.sqrt(std_x  ** 2 + std_y  ** 2 + std_z  ** 2 + 1e-8).astype(np.float32)

    mean_mag = mean_mag[:, np.newaxis]  # (300, 1)
    std_mag  = std_mag[:, np.newaxis]   # (300, 1)

    out = np.concatenate([sig[:, 0:6], mean_mag, std_mag], axis=1)  # (300, 8)
    return out.astype(np.float32)


# ──────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────

def load_all_samples(root_dir: str):
    """
    Walk *root_dir* and load every CSV file.

    Returns
    -------
    signals  : np.ndarray, shape (N, 300, 8)  – 8-channel feature array (NOT normalised)
    labels   : np.ndarray, shape (N,)          – original integer class index (0–5)
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

        # raw feature matrix (300, 6)
        sig_raw = df[FEATURE_COLS].values.astype(np.float32)
        assert sig_raw.shape == (SEQ_LEN, RAW_CHANNELS), \
            f"Unexpected shape {sig_raw.shape} for file {path}"

        # build 8-channel feature matrix
        sig = build_feature_channels(sig_raw)          # (300, 8)
        assert sig.shape == (SEQ_LEN, N_CHANNELS), \
            f"Unexpected post-feature shape {sig.shape} for file {path}"

        # label (one label per file, guaranteed consistent)
        lbl = int(df["label"].iloc[0])

        # group: derive integer user id from directory name
        user_dir  = os.path.basename(os.path.dirname(path))   # "User_042"
        user_id   = int(user_dir.split("_")[1])                # 42

        # file_id for submission
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
    flat = signals.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)
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
    StratifiedGroupKFold on the ORIGINAL labels (0–5), ensuring the same
    user never spans both splits.

    Returns
    -------
    train_idx, val_idx : 1-D integer arrays
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(sgkf.split(signals, labels, groups=groups))
    train_idx, val_idx = splits[fold_idx]
    return train_idx, val_idx


# ──────────────────────────────────────────
# Label remapping helpers
# ──────────────────────────────────────────

def remap_labels_stage1(labels: np.ndarray) -> np.ndarray:
    """
    Remap original 6-class labels to 5-class Stage 1 labels.
    L0→0, L1→1, L2→1, L3→2, L4→3, L5→4
    """
    out = np.empty_like(labels)
    for orig, mapped in STAGE1_MAP.items():
        out[labels == orig] = mapped
    return out


def filter_and_remap_stage2(signals: np.ndarray, labels: np.ndarray):
    """
    Filter to only L1/L2 samples and remap to binary labels.
    L1→0, L2→1

    Returns
    -------
    filtered_signals : np.ndarray, shape (M, 300, 8)
    filtered_labels  : np.ndarray, shape (M,)   – binary (0 or 1)
    mask             : np.ndarray, shape (N,)    – boolean mask used for filtering
    """
    mask = (labels == 1) | (labels == 2)
    filtered_signals = signals[mask]
    raw_labels       = labels[mask]
    filtered_labels  = np.array([STAGE2_MAP[int(l)] for l in raw_labels], dtype=np.int64)
    return filtered_signals, filtered_labels, mask


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
    time steps chosen uniformly at random.  signal: (300, 8)
    """
    signal = signal.copy()
    mask_len = np.random.randint(TIME_MASK_MIN, TIME_MASK_MAX + 1)
    start    = np.random.randint(0, SEQ_LEN - mask_len + 1)
    signal[start : start + mask_len, :] = 0.0
    return signal


def augment_channel_masking(signal: np.ndarray) -> np.ndarray:
    """
    Sensor Dropout – zero out CHANNEL_MASK_MIN to CHANNEL_MASK_MAX feature
    channels chosen uniformly at random.  signal: (300, 8)
    """
    signal   = signal.copy()
    n_mask   = np.random.randint(CHANNEL_MASK_MIN, CHANNEL_MASK_MAX + 1)
    channels = np.random.choice(N_CHANNELS, size=n_mask, replace=False)
    signal[:, channels] = 0.0
    return signal


def apply_augmentation(signal: np.ndarray, orig_label: int) -> np.ndarray:
    """
    Apply all augmentations independently with probability AUG_PROB each.
    Operates on signal of shape (300, 8) and returns the same shape.

    orig_label : the ORIGINAL label (0–5) before any mode remapping.
    Protected Exemption: Time Masking is skipped when orig_label == PROTECTED_ORIG_LABEL (2).
    """
    if orig_label != PROTECTED_ORIG_LABEL:
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
    PyTorch Dataset for HAR sequences supporting dynamic label modes.

    Parameters
    ----------
    signals      : np.ndarray, shape (N, 300, 8)  – already normalised
    labels       : np.ndarray, shape (N,)          – ALREADY remapped for the target mode
    orig_labels  : np.ndarray, shape (N,)          – original 0–5 labels (for augmentation protection)
    is_train     : bool  – enables global augmentation during training
    """

    def __init__(self,
                 signals:     np.ndarray,
                 labels:      np.ndarray,
                 orig_labels: np.ndarray,
                 is_train:    bool = False):
        self.signals     = signals.astype(np.float32)  # (N, 300, 8)
        self.labels      = labels.astype(np.int64)
        self.orig_labels = orig_labels.astype(np.int64)
        self.is_train    = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal     = self.signals[idx].copy()   # (300, 8)
        label      = self.labels[idx]
        orig_label = self.orig_labels[idx]

        if self.is_train:
            signal = apply_augmentation(signal, int(orig_label))

        # Transpose: (300, 8)  →  (8, 300)  for 1D-CNN input (C, L)
        signal = signal.T  # (8, 300)

        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


class HARTestDataset(Dataset):
    """
    Dataset for the held-out test set (no labels, no augmentation).

    Parameters
    ----------
    signals  : np.ndarray, shape (N, 300, 8)  – already normalised
    file_ids : np.ndarray, shape (N,)           – for submission mapping
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
# Convenience factories
# ──────────────────────────────────────────

def build_fold_datasets(train_root: str,
                        fold_idx:   int  = 0,
                        mode:       str  = "stage1"):
    """
    Full pipeline for one CV fold with dynamic label mode.

    Processing order:
      1. Load raw feature arrays → (N, 300, 8)  [mean_x/y/z + std_x/y/z + mean_mag + std_mag]
      2. Split into train/val using StratifiedGroupKFold on original labels
      3. Compute per-fold Z-score statistics from training fold only  (no leakage)
      4. Apply global Z-score normalisation to train and validation splits
      5. Apply label mode mapping (stage1 or stage2)

    Parameters
    ----------
    train_root : str    – path to training data directory
    fold_idx   : int    – 0-indexed fold number
    mode       : str    – 'stage1'  →  5-class generalist
                          'stage2'  →  2-class specialist (L1 vs L2 only)

    Returns
    -------
    train_dataset : HARDataset
    val_dataset   : HARDataset
    norm_params   : (mean, std) tuple – derived from training fold only
    class_counts  : np.ndarray  – per-class sample counts in the training split
                                   (indexed by remapped label)
    n_classes_out : int          – number of output classes for this mode
    """
    signals, labels, groups, _ = load_all_samples(train_root)

    train_idx, val_idx = get_fold_splits(signals, labels, groups, fold_idx)

    # Compute normalisation from training fold only (no leakage)
    mean, std = compute_normalization_params(signals[train_idx])

    train_signals_norm = normalize(signals[train_idx], mean, std)
    val_signals_norm   = normalize(signals[val_idx],   mean, std)

    train_labels_orig = labels[train_idx]
    val_labels_orig   = labels[val_idx]

    if mode == "stage1":
        train_labels_mapped = remap_labels_stage1(train_labels_orig)
        val_labels_mapped   = remap_labels_stage1(val_labels_orig)
        n_classes_out       = N_CLASSES_S1
        class_counts        = np.bincount(train_labels_mapped, minlength=N_CLASSES_S1)

        train_dataset = HARDataset(train_signals_norm, train_labels_mapped,
                                   train_labels_orig, is_train=True)
        val_dataset   = HARDataset(val_signals_norm,   val_labels_mapped,
                                   val_labels_orig,   is_train=False)

    elif mode == "stage2":
        # Filter to L1/L2 samples only in both splits
        train_sig_s2, train_lbl_s2, train_mask = filter_and_remap_stage2(
            train_signals_norm, train_labels_orig)
        val_sig_s2,   val_lbl_s2,   val_mask   = filter_and_remap_stage2(
            val_signals_norm,   val_labels_orig)

        train_orig_s2 = train_labels_orig[train_mask]
        val_orig_s2   = val_labels_orig[val_mask]

        n_classes_out = N_CLASSES_S2
        class_counts  = np.bincount(train_lbl_s2, minlength=N_CLASSES_S2)

        train_dataset = HARDataset(train_sig_s2, train_lbl_s2,
                                   train_orig_s2, is_train=True)
        val_dataset   = HARDataset(val_sig_s2,   val_lbl_s2,
                                   val_orig_s2,   is_train=False)

    elif mode == "flat":
        # v10: Flat 6-class global classifier – use original labels 0-5 directly,
        # no merging, no filtering.
        n_classes_out = N_CLASSES   # 6
        class_counts  = np.bincount(train_labels_orig, minlength=N_CLASSES)

        train_dataset = HARDataset(train_signals_norm, train_labels_orig,
                                   train_labels_orig, is_train=True)
        val_dataset   = HARDataset(val_signals_norm,   val_labels_orig,
                                   val_labels_orig,   is_train=False)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'stage1', 'stage2', or 'flat'.")

    return train_dataset, val_dataset, (mean, std), class_counts, n_classes_out


def load_test_samples(root_dir: str):
    """
    Walk *root_dir* and load every CSV file for the **test** set.
    Test CSVs have NO 'label' column: index, mean_x, …, std_z, file_id.

    Returns
    -------
    signals  : np.ndarray, shape (N, 300, 8)  – 8-channel array (NOT normalised)
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

        sig = build_feature_channels(sig_raw)   # (300, 8)

        file_id = int(df["file_id"].iloc[0])

        signals_list.append(sig)
        file_ids_list.append(file_id)

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 8)
    file_ids = np.array(file_ids_list, dtype=np.int64)

    return signals, file_ids


def build_test_dataset(test_root: str, norm_mean: np.ndarray, norm_std: np.ndarray):
    """
    Build the test dataset using normalisation parameters from the training fold.

    Processing order (v7):
      1. Load raw test signals → (N, 300, 8)
      2. Apply global Z-score normalisation using fold-derived mean/std
         (No instance centering – mirrors the v7 training pipeline)

    Returns
    -------
    test_dataset : HARTestDataset
    file_ids     : np.ndarray
    """
    signals, file_ids = load_test_samples(test_root)
    test_signals = normalize(signals, norm_mean, norm_std)
    return HARTestDataset(test_signals, file_ids), file_ids
