"""
dataset.py
----------
Data loading, preprocessing, normalization, and augmentation pipeline
for the HAR 1D-ResNet project.

Data format:
  - Each CSV: 300 rows × columns [index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id]
  - Feature shape after loading: (300, 6)  →  transposed to (6, 300) for 1D-CNN
  - Train users: User_001 – User_060
  - Test  users: User_061 – User_100
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
N_CHANNELS   = len(FEATURE_COLS)   # 6
SEQ_LEN      = 300
N_CLASSES    = 6
N_FOLDS      = 5

# Labels for which augmentation is applied during training
MINORITY_LABELS = {2, 3, 4, 5}

# Jitter noise std and scale range
JITTER_SIGMA = 0.05
SCALE_LOW    = 0.9
SCALE_HIGH   = 1.1


# ──────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────

def load_all_samples(root_dir: str):
    """
    Walk *root_dir* and load every CSV file.

    Returns
    -------
    signals : np.ndarray, shape (N, 300, 6)
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

        # --- feature matrix
        sig = df[FEATURE_COLS].values.astype(np.float32)  # (300, 6)
        assert sig.shape == (SEQ_LEN, N_CHANNELS), \
            f"Unexpected shape {sig.shape} for file {path}"

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

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 6)
    labels   = np.array(labels_list,   dtype=np.int64)
    groups   = np.array(groups_list,   dtype=np.int64)
    file_ids = np.array(file_ids_list, dtype=np.int64)

    return signals, labels, groups, file_ids


def compute_normalization_params(signals: np.ndarray):
    """
    Compute per-channel mean and std from a set of signals.

    Parameters
    ----------
    signals : np.ndarray, shape (N, 300, 6)

    Returns
    -------
    mean : np.ndarray, shape (6,)
    std  : np.ndarray, shape (6,)
    """
    # Reshape to (N*300, 6) then compute stats along axis 0
    flat = signals.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)   # avoid division by zero
    return mean, std


def normalize(signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization using *pre-computed* mean/std.
    signals : (N, 300, 6)  →  returns (N, 300, 6)
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
    """Add Gaussian noise to simulate sensor variance. signal: (300, 6)"""
    noise = np.random.normal(0.0, JITTER_SIGMA, size=signal.shape).astype(np.float32)
    return signal + noise


def augment_scale(signal: np.ndarray) -> np.ndarray:
    """Multiply by random per-channel scalar in [SCALE_LOW, SCALE_HIGH]. signal: (300, 6)"""
    scale = np.random.uniform(SCALE_LOW, SCALE_HIGH, size=(1, N_CHANNELS)).astype(np.float32)
    return signal * scale


def apply_augmentation(signal: np.ndarray) -> np.ndarray:
    """Apply all augmentations sequentially. signal: (300, 6)"""
    signal = augment_jitter(signal)
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
    signals    : np.ndarray, shape (N, 300, 6)  – already normalised
    labels     : np.ndarray, shape (N,)
    is_train   : bool  – enables augmentation for minority classes
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, is_train: bool = False):
        self.signals  = signals.astype(np.float32)  # (N, 300, 6)
        self.labels   = labels.astype(np.int64)
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()   # (300, 6)
        label  = self.labels[idx]

        # Apply augmentation ONLY during training AND for minority-class samples
        if self.is_train and int(label) in MINORITY_LABELS:
            signal = apply_augmentation(signal)

        # Transpose: (300, 6)  →  (6, 300)  for 1D-CNN input (C, L)
        signal = signal.T  # (6, 300)

        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


class HARTestDataset(Dataset):
    """
    Dataset for the held-out test set (no labels, no augmentation).

    Parameters
    ----------
    signals  : np.ndarray, shape (N, 300, 6)  – already normalised
    file_ids : np.ndarray, shape (N,)          – for submission mapping
    """

    def __init__(self, signals: np.ndarray, file_ids: np.ndarray):
        self.signals  = signals.astype(np.float32)
        self.file_ids = file_ids

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].T.copy()   # (6, 300)
        return torch.from_numpy(signal), self.file_ids[idx]


# ──────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────

def build_fold_datasets(train_root: str, fold_idx: int = 0):
    """
    Full pipeline for one CV fold.

    Returns
    -------
    train_dataset : HARDataset  (augmentation ON for minority classes)
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
    signals  : np.ndarray, shape (N, 300, 6)
    file_ids : np.ndarray, shape (N,)
    """
    csv_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found under: {root_dir}")

    signals_list  = []
    file_ids_list = []

    for path in csv_files:
        df = pd.read_csv(path)

        sig = df[FEATURE_COLS].values.astype(np.float32)  # (300, 6)
        assert sig.shape == (SEQ_LEN, N_CHANNELS), \
            f"Unexpected shape {sig.shape} for file {path}"

        file_id = int(df["file_id"].iloc[0])

        signals_list.append(sig)
        file_ids_list.append(file_id)

    signals  = np.stack(signals_list,  axis=0)          # (N, 300, 6)
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
