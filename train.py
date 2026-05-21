"""
train.py
--------
Hierarchical Two-Stage Training Pipeline for HAR InceptionTime (v9).

Architecture
────────────
  Stage 1 (Generalist)  : 5-class InceptionTime1D – classifies L0, L1_2_Merged, L3, L4, L5
  Stage 2 (Specialist)  : Heterogeneous Dual-Engine Ensemble
                            Engine A – Binary InceptionTime1D (L1 vs L2, BCEWithLogitsLoss)
                            Engine B – GBDT (LightGBM) on handcrafted tabular features

Training Phases
───────────────
  Phase A – Stage 1 Generalist:
    • 5-fold StratifiedGroupKFold on merged labels
    • Uniform Focal Loss (γ=2.0, ε=0.1 label smoothing, α=inverse-frequency)
    • Batch-level Mixup (Beta(α=0.2, α=0.2))
    • AdamW + CosineAnnealingLR
    • Early stop on validation Macro F1 (patience=25)
    • Saves: checkpoints/stage1_fold_{n}_best.pth

  Phase B – Stage 2 Specialist (L1 vs L2):
    Engine A (Neural Network):
    • 5-fold CV on L1/L2 samples only
    • BCEWithLogitsLoss with computed pos_weight = count(L1) / count(L2)
    • AdamW + CosineAnnealingLR
    • Early stop on validation Macro F1 (patience=25)
    • Saves: checkpoints/stage2_fold_{n}_best.pth

    Engine B (GBDT):
    • Extracts tabular statistical features from std_mag channel (channel index 7):
        10th / 25th / 75th / 90th percentiles, Kurtosis, Skewness, Zero-crossing rate
    • Also extracts mean/std summaries across all 8 channels
    • Trains one LightGBM binary classifier per CV fold

Inference (Hierarchical Routing)
─────────────────────────────────
  1. Run all test samples through Stage 1 ensemble (5 folds × TTA ×3 scales)
     → F1-weighted softmax averaging → argmax → 5-class prediction
  2. If predicted class == 1 (L1_2_Merged) → route to Stage 2
     Else → accept Stage 1 prediction; map back to original label space
  3. Stage 2 dual-engine inference on routed samples:
        P_CNN  = Engine A InceptionTime sigmoid probabilities (F1-weighted TTA)
        P_GBDT = Engine B LightGBM probability of L2
        P_final = 0.5 × P_CNN + 0.5 × P_GBDT
        threshold → L1 or L2
  4. Recombine and write: submission_v9_inception.csv

Test-Time Feature Normalization (Domain Adaptation)
────────────────────────────────────────────────────
  Test data is NOT normalised with training global mean/std.
  Instead the test set's own batch mean and std are computed at inference
  time and used for intrinsic Z-score standardisation, resolving
  cross-subject covariate shift without any labels.

Usage
─────
    conda activate PyTorch
    python train.py               # fold 1 only (Phase A + Phase B + inference)
    python train.py --all-folds   # full 5-fold CV + ensemble inference

DO NOT run this script automatically – see task constraints.
"""

import os
import sys
import time
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

if not LGBM_AVAILABLE:
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
        raise ImportError(
            "Neither LightGBM nor XGBoost is installed. "
            "Please install one: pip install lightgbm  OR  pip install xgboost")
else:
    XGB_AVAILABLE = False

from dataset import (
    build_fold_datasets, build_test_dataset,
    load_all_samples, load_test_samples, get_fold_splits,
    normalize, remap_labels_stage1, HARDataset,
    N_FOLDS, N_CLASSES, N_CLASSES_S1, N_CLASSES_S2,
    STAGE1_MAP, STAGE2_MAP,
)
from model import InceptionTime1D

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# Module 1: Deterministic Experimental Control
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42) -> None:
    """Lock all random number generators for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# Apply global seed immediately at module load time
seed_everything(42)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Paths ───────────────────────────────────────────────────────────────────
    "train_root"        : "./train/train",
    "test_root"         : "./test/test",
    "submission_path"   : "./submission_v9_inception.csv",
    "checkpoint_dir"    : "./checkpoints",
    "log_path"          : "./training_log.txt",

    # Training (shared) ───────────────────────────────────────────────────────
    "epochs"            : 80,
    "batch_size"        : 128,
    "lr"                : 3e-4,
    "weight_decay"      : 1e-3,
    "num_workers"       : 4,
    "pin_memory"        : True,

    # CosineAnnealingLR
    "T_max"             : 80,
    "eta_min"           : 1e-6,

    # Early stopping ──────────────────────────────────────────────────────────
    "patience"          : 25,

    # Model (InceptionTime) ───────────────────────────────────────────────────
    "in_channels"       : 8,       # v9: 8 stable channels
    "nb_filters"        : 32,      # output channels per branch per block
    "bottleneck"        : 32,      # bottleneck projection dim
    "depth"             : 6,       # number of stacked Inception blocks

    # Stage-specific class counts
    "num_classes_s1"    : N_CLASSES_S1,   # 5
    "num_classes_s2"    : N_CLASSES_S2,   # 2  (BCE with logits → 1 output logit)

    # Focal Loss (Stage 1)
    "focal_gamma"       : 2.0,
    "label_smoothing"   : 0.1,

    # Mixup (Stage 1 only)
    "mixup_alpha"       : 0.2,

    # Stage 2 decision threshold (parameterized for micro-tuning)
    "s2_threshold"      : 0.5,

    # Cross-validation ────────────────────────────────────────────────────────
    "run_all_folds"     : False,

    # GBDT Engine B weights for ensemble (0.5 each)
    "gbdt_weight"       : 0.5,
    "cnn_weight"        : 0.5,

    # Reproducibility ─────────────────────────────────────────────────────────
    "seed"              : 42,
}


# ══════════════════════════════════════════════════════════════════════════════
# Dual-stream logger
# ══════════════════════════════════════════════════════════════════════════════

class DualLogger:
    """Routes output to both the terminal and a persistent text file."""

    def __init__(self, log_path: str):
        log_dir = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(log_dir, exist_ok=True)
        self._fh      = open(log_path, "w", encoding="utf-8", buffering=1)
        self._console = sys.__stdout__

    def log(self, msg: str = ""):
        self._console.write(msg + "\n")
        self._console.flush()
        self._fh.write(msg + "\n")

    def log_file(self, msg: str = ""):
        self._fh.write(msg + "\n")

    def log_console(self, msg: str = ""):
        self._console.write(msg + "\n")
        self._console.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Module 2: Focal Loss (Stage 1 – 5-class generalist)
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Alpha-weighted Focal Loss with uniform γ and label smoothing ε.

    Supports both integer hard-label targets (1-D) and pre-smoothed
    soft-target distributions (2-D, e.g., from Mixup).
    """

    def __init__(self, alpha: torch.Tensor,
                 gamma: float           = 2.0,
                 label_smoothing: float = 0.1,
                 reduction: str         = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(1)

        if targets.dim() == 1:
            with torch.no_grad():
                smooth_val = self.label_smoothing / C
                y_smooth   = torch.full_like(logits, smooth_val)
                y_smooth.scatter_(1, targets.unsqueeze(1),
                                  1.0 - self.label_smoothing + smooth_val)
        else:
            y_smooth = targets

        log_probs  = F.log_softmax(logits, dim=1)
        probs      = log_probs.exp()
        true_class = y_smooth.argmax(dim=1)
        pt         = probs.gather(1, true_class.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        alpha_t      = self.alpha.gather(0, true_class)
        ce_smooth    = -(y_smooth * log_probs).sum(dim=1)
        loss         = alpha_t * focal_weight * ce_smooth

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_focal_loss(class_counts: np.ndarray,
                     num_classes:  int,
                     device:       torch.device,
                     gamma:        float = 2.0,
                     label_smoothing: float = 0.1) -> FocalLoss:
    """Construct FocalLoss with inverse-frequency alpha weights."""
    counts  = class_counts.astype(np.float32)
    total   = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.mean()
    alpha   = torch.tensor(weights, dtype=torch.float32, device=device)
    return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)


# ══════════════════════════════════════════════════════════════════════════════
# Module 3: Mixup augmentation (Stage 1 only)
# ══════════════════════════════════════════════════════════════════════════════

def mixup_batch(signals:       torch.Tensor,
                labels:        torch.Tensor,
                alpha:         float,
                num_classes:   int,
                label_smoothing: float,
                device:        torch.device):
    """
    Batch-level Mixup with label smoothing.

    Returns x_mixed (B, C, L), y_mixed (B, num_classes), lam.
    """
    B = signals.size(0)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(B, device=device)

    smooth_val = label_smoothing / num_classes
    y_i = torch.full((B, num_classes), smooth_val, dtype=torch.float32, device=device)
    y_i.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing + smooth_val)
    y_j = y_i[perm]

    x_mixed = lam * signals + (1.0 - lam) * signals[perm]
    y_mixed = lam * y_i     + (1.0 - lam) * y_j

    return x_mixed, y_mixed, lam


# ══════════════════════════════════════════════════════════════════════════════
# Early stopping helper
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4,
                 checkpoint_path: str = "best_model.pth"):
        self.patience        = patience
        self.min_delta       = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score      = -np.inf
        self.counter         = 0
        self.best_epoch      = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
# Module 4: Tabular Feature Extractor (Stage 2 – Engine B)
# ══════════════════════════════════════════════════════════════════════════════

# Channel index for std_mag (the 8th channel, 0-indexed = 7)
STD_MAG_CHANNEL = 7


def zero_crossing_rate(signal_1d: np.ndarray) -> float:
    """Compute zero-crossing rate for a 1-D signal of length L."""
    signs = np.sign(signal_1d)
    # Replace 0 sign with previous sign to avoid ambiguity
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings) / max(len(signal_1d) - 1, 1)


def extract_tabular_features(signals_np: np.ndarray) -> np.ndarray:
    """
    Extract a handcrafted tabular feature vector from a batch of
    raw 300-timestep signal windows.

    Parameters
    ----------
    signals_np : np.ndarray, shape (N, 8, 300)
        The 8-channel × 300-timestep windows.

    Returns
    -------
    features : np.ndarray, shape (N, n_features)

    Feature layout
    ──────────────
    std_mag channel (channel 7) – 7 features:
        p10, p25, p75, p90  (4 percentiles)
        kurtosis            (1)
        skewness            (1)
        zero_crossing_rate  (1)

    All 8 channels – per-channel mean + per-channel std  (8×2 = 16 features)

    Total: 7 + 16 = 23 features per sample.
    """
    N = signals_np.shape[0]
    feature_rows = []

    for i in range(N):
        sample = signals_np[i]           # (8, 300)
        std_mag = sample[STD_MAG_CHANNEL]  # (300,)

        # ── std_mag-specific features ────────────────────────────────────────
        p10 = float(np.percentile(std_mag, 10))
        p25 = float(np.percentile(std_mag, 25))
        p75 = float(np.percentile(std_mag, 75))
        p90 = float(np.percentile(std_mag, 90))
        kurt = float(scipy_kurtosis(std_mag, fisher=True, bias=True))
        skew = float(scipy_skew(std_mag, bias=True))
        zcr  = zero_crossing_rate(std_mag)

        # ── All-channel summary features ────────────────────────────────────
        ch_mean = sample.mean(axis=1)   # (8,)
        ch_std  = sample.std(axis=1)    # (8,)

        row = np.array([p10, p25, p75, p90, kurt, skew, zcr,
                        *ch_mean.tolist(), *ch_std.tolist()],
                       dtype=np.float32)
        feature_rows.append(row)

    return np.stack(feature_rows, axis=0)   # (N, 23)


def build_gbdt_model():
    """
    Instantiate and return a LightGBM (preferred) or XGBoost binary classifier
    with sane defaults for the L1/L2 imbalanced task.
    """
    if LGBM_AVAILABLE:
        model = lgb.LGBMClassifier(
            n_estimators    = 500,
            learning_rate   = 0.05,
            max_depth       = 6,
            num_leaves      = 31,
            min_child_samples= 20,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            reg_alpha       = 0.1,
            reg_lambda      = 1.0,
            random_state    = 42,
            verbose         = -1,
            n_jobs          = -1,
        )
        backend = "LightGBM"
    else:
        model = xgb.XGBClassifier(
            n_estimators    = 500,
            learning_rate   = 0.05,
            max_depth       = 6,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            reg_alpha       = 0.1,
            reg_lambda      = 1.0,
            use_label_encoder= False,
            eval_metric     = "logloss",
            random_state    = 42,
            verbosity       = 0,
            n_jobs          = -1,
        )
        backend = "XGBoost"
    return model, backend


# ══════════════════════════════════════════════════════════════════════════════
# Phase A helpers – Stage 1 (5-class generalist, Focal Loss + Mixup)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch_s1(model, loader, criterion, optimizer, device,
                       mixup_alpha: float, label_smoothing: float,
                       num_classes: int) -> float:
    """Train Stage 1 for one epoch with Mixup + Focal Loss."""
    model.train()
    running_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)   # (B, 8, 300)
        labels  = labels.to(device,  non_blocking=True)   # (B,)

        x_mixed, y_mixed, _ = mixup_batch(
            signals, labels,
            alpha=mixup_alpha,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            device=device,
        )

        optimizer.zero_grad()
        logits = model(x_mixed)
        loss   = criterion(logits, y_mixed)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * signals.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_s1(model, loader, criterion, device):
    """Validation for Stage 1 (multiclass, hard labels)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        logits  = model(signals)
        running_loss += criterion(logits, labels).item() * signals.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Phase B helpers – Stage 2 (binary specialist, BCEWithLogitsLoss)
# ══════════════════════════════════════════════════════════════════════════════

def build_bce_loss(class_counts: np.ndarray, device: torch.device) -> nn.BCEWithLogitsLoss:
    """
    Construct BCEWithLogitsLoss with computed pos_weight.

    pos_weight = count(class 0 = L1) / count(class 1 = L2)
    This scales up the loss for L2 samples if L1 is more frequent.
    """
    count_neg = float(class_counts[0])   # L1 mapped to binary 0
    count_pos = float(class_counts[1])   # L2 mapped to binary 1
    if count_pos < 1.0:
        count_pos = 1.0
    pos_weight = torch.tensor([count_neg / count_pos],
                               dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch_s2(model, loader, criterion, optimizer, device) -> float:
    """
    Train Stage 2 Engine A (InceptionTime) for one epoch.
    model(signals) returns (B, 1) logit tensor.
    """
    model.train()
    running_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)          # (B, 8, 300)
        labels  = labels.to(device,  non_blocking=True).float()  # (B,)

        optimizer.zero_grad()
        output = model(signals)   # (B, 1)
        loss   = criterion(output.squeeze(-1), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * signals.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_s2(model, loader, criterion, device, threshold: float = 0.5):
    """
    Validation for Stage 2 Engine A (binary, BCEWithLogitsLoss).
    model.eval() → InceptionTime1D returns (B, 1) logit.
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True).float()
        logits  = model(signals).squeeze(-1)   # (B,)
        running_loss += criterion(logits, labels).item() * signals.size(0)
        preds = (torch.sigmoid(logits) >= threshold).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.long().cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Per-fold Stage 1 training
# ══════════════════════════════════════════════════════════════════════════════

def train_fold_s1(fold_idx: int, device: torch.device, cfg: dict,
                  logger: DualLogger):
    """
    Train one CV fold for Stage 1 (5-class generalist).

    Returns
    -------
    best_val_f1     : float
    norm_params     : (mean, std)
    checkpoint_path : str
    best_val_report : str
    best_val_preds  : (val_preds, val_labels)
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase A] --- Fold {fold_num}/{N_FOLDS} Stage1 Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE A | FOLD {fold_num} / {N_FOLDS}  – Stage 1 Generalist (5-class InceptionTime)")
    logger.log_file("=" * 70)

    train_ds, val_ds, norm_params, class_counts, n_cls = build_fold_datasets(
        cfg["train_root"], fold_idx=fold_idx, mode="stage1"
    )

    logger.log_file(f"  Training samples  : {len(train_ds)}")
    logger.log_file(f"  Validation samples: {len(val_ds)}")
    logger.log_file("  Stage1 class counts (train fold):")
    s1_label_names = ["L0", "L1_2_Merged", "L3", "L4", "L5"]
    for c, cnt in enumerate(class_counts):
        logger.log_file(f"    {s1_label_names[c]}: {cnt:>5d}  ({cnt / class_counts.sum() * 100:.1f}%)")

    logger.log_console(f"  [S1] Train: {len(train_ds)}  Val: {len(val_ds)}  "
                       f"Classes: {class_counts.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    model_s1 = InceptionTime1D(
        in_channels = cfg["in_channels"],
        num_classes = cfg["num_classes_s1"],
        nb_filters  = cfg["nb_filters"],
        bottleneck  = cfg["bottleneck"],
        depth       = cfg["depth"],
    ).to(device)
    n_params = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    logger.log_file(f"  Model_Stage1: InceptionTime1D (nb_filters={cfg['nb_filters']}, "
                    f"depth={cfg['depth']})  |  Params: {n_params:,}")

    criterion_s1 = build_focal_loss(
        class_counts, cfg["num_classes_s1"], device,
        gamma=cfg["focal_gamma"],
        label_smoothing=cfg["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(model_s1.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path  = os.path.join(cfg["checkpoint_dir"], f"stage1_fold_{fold_num}_best.pth")
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    best_val_preds  = (None, None)
    best_val_report = ""

    for epoch in range(1, cfg["epochs"] + 1):
        t_start    = time.time()
        train_loss = train_one_epoch_s1(
            model_s1, train_loader, criterion_s1, optimizer, device,
            mixup_alpha=cfg["mixup_alpha"],
            label_smoothing=cfg["label_smoothing"],
            num_classes=cfg["num_classes_s1"],
        )
        scheduler.step()
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate_s1(
            model_s1, val_loader, criterion_s1, device)
        elapsed = time.time() - t_start
        lr_now  = scheduler.get_last_lr()[0]

        logger.log_console(
            f"[S1|Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.1e} | "
            f"Loss: T:{train_loss:.2f} / V:{val_loss:.2f} | "
            f"Val Acc: {val_acc:5.2f}% | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        logger.log_file(
            f"[S1|Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.4e} | "
            f"Loss: T:{train_loss:.6f} / V:{val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f}% | "
            f"Val F1: {val_f1:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        if val_f1 >= early_stop.best_score:
            best_val_preds  = (val_preds, val_labels)
            best_val_report = classification_report(
                val_labels, val_preds,
                target_names=s1_label_names, digits=4, zero_division=0)

        should_stop = early_stop.step(val_f1, model_s1, epoch)
        if should_stop:
            msg = (f"  >> [S1|Fold {fold_num}] Early stop @ epoch {epoch}. "
                   f"Best epoch: {early_stop.best_epoch}  "
                   f"Best F1: {early_stop.best_score:.4f}")
            logger.log(msg)
            break

    fold_summary = (
        f"--- [S1] Fold {fold_num} done | "
        f"Best Val Macro F1: {early_stop.best_score:.6f} "
        f"(epoch {early_stop.best_epoch}) ---"
    )
    logger.log(fold_summary)
    logger.log_file("")
    logger.log_file(f"  [S1] Classification Report @ Best Epoch ({early_stop.best_epoch}):")
    logger.log_file(best_val_report)
    logger.log_file("─" * 70)

    return early_stop.best_score, norm_params, ckpt_path, best_val_report, best_val_preds


# ══════════════════════════════════════════════════════════════════════════════
# Phase B: Per-fold Stage 2 training (Dual-Engine: InceptionTime + GBDT)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_raw_signals_from_loader(loader) -> np.ndarray:
    """
    Iterate a DataLoader and return the concatenated raw signal arrays.

    DataLoader yields (signals_tensor, labels_tensor) where
    signals_tensor is (B, 8, 300).  Returns np.ndarray (N, 8, 300).
    """
    all_signals = []
    all_labels  = []
    for signals, labels in loader:
        all_signals.append(signals.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_signals, axis=0), np.concatenate(all_labels, axis=0)


def train_fold_s2(fold_idx: int, device: torch.device, cfg: dict,
                  logger: DualLogger):
    """
    Train one CV fold for Stage 2.

    Engine A: InceptionTime1D (BCEWithLogitsLoss)
    Engine B: LightGBM / XGBoost on handcrafted tabular features

    Returns
    -------
    best_val_f1      : float  (Engine A best validation F1)
    checkpoint_path  : str    (Engine A checkpoint)
    gbdt_model       : fitted GBDT model for this fold
    best_val_report  : str
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase B] --- Fold {fold_num}/{N_FOLDS} Stage2 Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE B | FOLD {fold_num} / {N_FOLDS}  – Stage 2 Specialist "
                    f"(L1 vs L2 | InceptionTime + GBDT)")
    logger.log_file("=" * 70)

    train_ds, val_ds, _, class_counts, n_cls = build_fold_datasets(
        cfg["train_root"], fold_idx=fold_idx, mode="stage2"
    )

    logger.log_file(f"  Training samples  (L1+L2 only): {len(train_ds)}")
    logger.log_file(f"  Validation samples(L1+L2 only): {len(val_ds)}")
    logger.log_file("  Stage2 class counts (train fold):")
    s2_label_names = ["L1", "L2"]
    for c, cnt in enumerate(class_counts):
        logger.log_file(f"    {s2_label_names[c]}: {cnt:>5d}  ({cnt / max(class_counts.sum(), 1) * 100:.1f}%)")

    logger.log_console(f"  [S2] Train: {len(train_ds)}  Val: {len(val_ds)}  "
                       f"Classes: {class_counts.tolist()}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.log(f"  [S2|Fold {fold_num}] WARNING: No L1/L2 samples in this split – skipping.")
        ckpt_path = os.path.join(cfg["checkpoint_dir"], f"stage2_fold_{fold_num}_best.pth")
        return 0.0, ckpt_path, None, ""

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    # ── Engine A: InceptionTime1D ─────────────────────────────────────────────
    model_s2 = InceptionTime1D(
        in_channels = cfg["in_channels"],
        num_classes = 1,
        nb_filters  = cfg["nb_filters"],
        bottleneck  = cfg["bottleneck"],
        depth       = cfg["depth"],
    ).to(device)
    n_params = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    logger.log_file(f"  Engine A: InceptionTime1D (nb_filters={cfg['nb_filters']}, "
                    f"depth={cfg['depth']})  |  Params: {n_params:,}")

    criterion_s2 = build_bce_loss(class_counts, device)
    pos_w = float(class_counts[0]) / max(float(class_counts[1]), 1.0)
    logger.log_file(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_w:.4f}  "
                    f"[count_L1={class_counts[0]}, count_L2={class_counts[1]}])")

    optimizer = torch.optim.AdamW(model_s2.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path  = os.path.join(cfg["checkpoint_dir"], f"stage2_fold_{fold_num}_best.pth")
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    best_val_preds  = (None, None)
    best_val_report = ""

    for epoch in range(1, cfg["epochs"] + 1):
        t_start    = time.time()
        train_loss = train_one_epoch_s2(model_s2, train_loader, criterion_s2,
                                        optimizer, device)
        scheduler.step()
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate_s2(
            model_s2, val_loader, criterion_s2, device,
            threshold=cfg["s2_threshold"])
        elapsed = time.time() - t_start
        lr_now  = scheduler.get_last_lr()[0]

        logger.log_console(
            f"[S2|Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.1e} | "
            f"Loss: T:{train_loss:.2f} / V:{val_loss:.2f} | "
            f"Val Acc: {val_acc:5.2f}% | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        logger.log_file(
            f"[S2|Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.4e} | "
            f"Loss: T:{train_loss:.6f} / V:{val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f}% | "
            f"Val F1: {val_f1:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        if val_f1 >= early_stop.best_score:
            best_val_preds  = (val_preds, val_labels)
            best_val_report = classification_report(
                val_labels, val_preds,
                target_names=s2_label_names, digits=4, zero_division=0)

        should_stop = early_stop.step(val_f1, model_s2, epoch)
        if should_stop:
            msg = (f"  >> [S2|Fold {fold_num}] Early stop @ epoch {epoch}. "
                   f"Best epoch: {early_stop.best_epoch}  "
                   f"Best F1: {early_stop.best_score:.4f}")
            logger.log(msg)
            break

    fold_summary = (
        f"--- [S2 Engine A] Fold {fold_num} done | "
        f"Best Val Macro F1: {early_stop.best_score:.6f} "
        f"(epoch {early_stop.best_epoch}) ---"
    )
    logger.log(fold_summary)
    logger.log_file("")
    logger.log_file(f"  [S2 Engine A] Classification Report @ Best Epoch ({early_stop.best_epoch}):")
    logger.log_file(best_val_report)
    logger.log_file("─" * 70)

    # ── Engine B: GBDT on tabular features ───────────────────────────────────
    logger.log(f"\n  [S2 Engine B|Fold {fold_num}] Extracting tabular features for GBDT ...")

    # Re-iterate loaders with shuffle=False to get ordered arrays
    train_loader_ord = DataLoader(train_ds, batch_size=cfg["batch_size"] * 2,
                                  shuffle=False, num_workers=cfg["num_workers"],
                                  pin_memory=False)
    val_loader_ord   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                                  shuffle=False, num_workers=cfg["num_workers"],
                                  pin_memory=False)

    train_signals_np, train_labels_np = _collect_raw_signals_from_loader(train_loader_ord)
    val_signals_np,   val_labels_np   = _collect_raw_signals_from_loader(val_loader_ord)

    X_train_tab = extract_tabular_features(train_signals_np)   # (N_train, 23)
    X_val_tab   = extract_tabular_features(val_signals_np)     # (N_val,   23)

    sample_weight = None
    if class_counts[1] > 0:
        # Inverse-frequency sample weights to counter 13:1 imbalance
        ratio = float(class_counts[0]) / float(class_counts[1])
        sample_weight = np.where(train_labels_np == 1, ratio, 1.0).astype(np.float32)

    gbdt_model, backend = build_gbdt_model()
    logger.log(f"  [S2 Engine B|Fold {fold_num}] Training {backend} on "
               f"{len(train_labels_np)} samples ({X_train_tab.shape[1]} features) ...")

    t_gbdt = time.time()
    if sample_weight is not None:
        gbdt_model.fit(X_train_tab, train_labels_np, sample_weight=sample_weight)
    else:
        gbdt_model.fit(X_train_tab, train_labels_np)
    gbdt_elapsed = time.time() - t_gbdt

    gbdt_val_preds = gbdt_model.predict(X_val_tab)
    gbdt_val_f1    = f1_score(val_labels_np, gbdt_val_preds, average="macro", zero_division=0)
    gbdt_val_acc   = accuracy_score(val_labels_np, gbdt_val_preds) * 100.0
    logger.log(f"  [S2 Engine B|Fold {fold_num}] {backend} Val Macro F1: {gbdt_val_f1:.4f}  "
               f"Acc: {gbdt_val_acc:.2f}%  Time: {gbdt_elapsed:.1f}s")
    logger.log_file(f"  [S2 Engine B|Fold {fold_num}] {backend} Val Macro F1: {gbdt_val_f1:.6f}  "
                    f"Acc: {gbdt_val_acc:.4f}%  Time: {gbdt_elapsed:.2f}s")
    logger.log_file("─" * 70)

    return early_stop.best_score, ckpt_path, gbdt_model, best_val_report


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix formatting helper
# ══════════════════════════════════════════════════════════════════════════════

def format_confusion_matrix(all_labels, all_preds, label_names=None) -> str:
    n_cls   = len(np.unique(np.concatenate([all_labels, all_preds])))
    labels  = list(range(n_cls))
    cm      = confusion_matrix(all_labels, all_preds, labels=labels)
    if label_names is None:
        label_names = [str(i) for i in labels]
    header  = "Pred→  " + "  ".join(f"  {ln[:4]:>4}" for ln in label_names)
    divider = "─" * len(header)
    rows    = [header, divider]
    for i, row in enumerate(cm):
        rows.append(f"True {label_names[i][:4]:>4} | " + " | ".join(f"{v:5d}" for v in row))
    rows.append("(Rows = True Label, Columns = Predicted Label)")
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Module 5: Hierarchical Dual-Engine Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def hierarchical_inference(s1_fold_results: list,
                            s2_fold_results: list,
                            cfg:             dict,
                            device:          torch.device,
                            logger:          DualLogger) -> pd.DataFrame:
    """
    Full hierarchical routing inference pipeline (v9).

    Step 1: Stage 1 TTA ensemble (5 folds × 3 TTA scales, F1-weighted avg)
    Step 2: argmax → accept non-merged predictions, flag merged (idx==1)
    Step 3: Stage 2 dual-engine on flagged samples:
               Engine A: InceptionTime CNN sigmoid (F1-weighted TTA)
               Engine B: GBDT probability on tabular features
               P_final = (cnn_weight × P_CNN) + (gbdt_weight × P_GBDT)
    Step 4: Recombine and write submission_v9_inception.csv

    Test-Time Feature Normalization (Domain Adaptation):
        Test data is normalised using the test set's OWN mean and std,
        not the training distribution parameters. This corrects cross-subject
        covariate shift in an unsupervised manner.

    Parameters
    ----------
    s1_fold_results : list of (fold_idx, best_f1, norm_params, ckpt_path, report)
    s2_fold_results : list of (fold_idx, best_f1, ckpt_path, gbdt_model, report)
    cfg             : dict
    device          : torch.device
    logger          : DualLogger

    Returns
    -------
    submission : pd.DataFrame  with columns ['Id', 'Label']
    """
    logger.log("\n" + "=" * 70)
    logger.log("  HIERARCHICAL INFERENCE  (Stage1 → Route → Stage2 Dual-Engine)")
    logger.log("=" * 70)

    # ── Test-Time Feature Normalization ──────────────────────────────────────
    # Load raw test signals (NOT normalised with training stats) using
    # load_test_samples, which returns (N, 300, 8) without any normalisation.
    # Then compute test-batch mean/std → intrinsic Z-score normalisation.
    logger.log("\n  [TTFN] Loading test data with intrinsic Z-score normalisation ...")

    # load_test_samples returns: signals (N, 300, 8), file_ids (N,)
    raw_test_signals_300_8, file_ids = load_test_samples(cfg["test_root"])

    # Transpose to (N, 8, 300) to match CNN input convention
    raw_test_signals = raw_test_signals_300_8.transpose(0, 2, 1).astype(np.float32)  # (N, 8, 300)
    n_test = raw_test_signals.shape[0]

    # Compute per-channel mean and std across the test batch
    # raw_test_signals: (N, 8, 300) → flatten per channel → (N*300,) per channel
    test_flat = raw_test_signals.transpose(1, 0, 2).reshape(
        raw_test_signals.shape[1], -1)   # (8, N*300)
    test_mean = test_flat.mean(axis=1, keepdims=True)   # (8, 1)
    test_std  = test_flat.std(axis=1,  keepdims=True)   # (8, 1)
    test_std  = np.where(test_std < 1e-8, 1.0, test_std)

    # Apply intrinsic standardisation: (N, 8, 300)
    # normalise along channel dim: broadcast (8,1) → (8,300) → (N,8,300)
    normed_test = (raw_test_signals - test_mean[np.newaxis, :, :]) / \
                   test_std[np.newaxis, :, :]
    normed_test = normed_test.astype(np.float32)

    logger.log(f"  [TTFN] Test batch stats computed independently (N={n_test}). "
               f"Per-channel mean: {test_mean.squeeze().tolist()}")
    logger.log(f"  [TTFN] Per-channel std : {test_std.squeeze().tolist()}")

    # Build a TensorDataset from the normalised test signals
    normed_tensor = torch.from_numpy(normed_test)   # (N, 8, 300)
    # Dummy label tensor (inference only)
    dummy_labels  = torch.zeros(n_test, dtype=torch.long)
    normed_test_ds     = TensorDataset(normed_tensor, dummy_labels)
    normed_test_loader = DataLoader(
        normed_test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    logger.log(f"  Total test samples: {n_test}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Stage 1 TTA ensemble – accumulate 5-class probability vectors
    # ══════════════════════════════════════════════════════════════════════════
    logger.log("\n  [Step 1] Stage 1 Generalist Ensemble ...")

    s1_weighted_prob_sum = np.zeros((n_test, cfg["num_classes_s1"]), dtype=np.float64)
    s1_f1_total          = 0.0

    for fold_idx, best_f1, norm_params, ckpt_path, _ in s1_fold_results:
        fold_num = fold_idx + 1
        logger.log(f"    S1 Fold {fold_num}: loading {ckpt_path}  (Val F1={best_f1:.6f})")

        model_s1 = InceptionTime1D(
            in_channels = cfg["in_channels"],
            num_classes = cfg["num_classes_s1"],
            nb_filters  = cfg["nb_filters"],
            bottleneck  = cfg["bottleneck"],
            depth       = cfg["depth"],
        ).to(device)
        model_s1.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_s1.eval()

        fold_probs = []
        for signals, _ in normed_test_loader:
            x = signals.to(device, non_blocking=True)
            # TTA: original / scale-up (+5%) / scale-down (−5%)
            p_orig = F.softmax(model_s1(x),        dim=1)
            p_up   = F.softmax(model_s1(x * 1.05), dim=1)
            p_down = F.softmax(model_s1(x * 0.95), dim=1)
            p_tta  = (p_orig + p_up + p_down) / 3.0
            fold_probs.append(p_tta.cpu().numpy())

        fold_probs_arr        = np.concatenate(fold_probs, axis=0)   # (N, 5)
        s1_weighted_prob_sum += best_f1 * fold_probs_arr
        s1_f1_total          += best_f1
        logger.log(f"      → accumulated (weight={best_f1:.6f})")

    # F1-weighted ensemble probability for Stage 1
    s1_ensemble_probs = s1_weighted_prob_sum / s1_f1_total   # (N, 5)
    s1_preds          = s1_ensemble_probs.argmax(axis=1)     # (N,)  in [0..4]

    # ── Map Stage 1 predictions back to original label space ─────────────────
    # Stage 1 index → original label:
    #   0 → L0,  1 → L1_2_Merged (routed to S2),  2 → L3,  3 → L4,  4 → L5
    STAGE1_IDX_TO_ORIG = {0: 0, 1: -1, 2: 3, 3: 4, 4: 5}   # -1 = route to S2

    # Indices of samples routed to Stage 2 (Stage 1 predicted "merged" class)
    s2_route_mask = (s1_preds == 1)
    n_routed      = int(s2_route_mask.sum())
    n_resolved    = int((~s2_route_mask).sum())

    s1_dist = {idx: int((s1_preds == idx).sum()) for idx in range(cfg["num_classes_s1"])}
    logger.log(f"\n  Stage 1 prediction distribution: {s1_dist}")
    logger.log(f"  Samples resolved by Stage 1   : {n_resolved}")
    logger.log(f"  Samples routed to Stage 2     : {n_routed}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 + 3: Stage 2 Dual-Engine Specialist for routed samples
    # ══════════════════════════════════════════════════════════════════════════
    # Final predictions array in ORIGINAL label space (0–5)
    final_preds = np.full(n_test, -1, dtype=np.int64)

    # Accept non-merged Stage 1 predictions
    for s1_idx, orig_lbl in STAGE1_IDX_TO_ORIG.items():
        if orig_lbl == -1:
            continue
        mask = (s1_preds == s1_idx)
        final_preds[mask] = orig_lbl

    if n_routed > 0:
        logger.log("\n  [Step 3] Stage 2 Dual-Engine for routed samples ...")

        # Extract routed samples from the normalised test tensor
        routed_signals_np = normed_test[s2_route_mask]          # (n_routed, 8, 300)
        routed_tensor     = torch.from_numpy(routed_signals_np)  # (n_routed, 8, 300)
        routed_ds         = TensorDataset(routed_tensor)
        routed_loader     = DataLoader(
            routed_ds,
            batch_size=cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=0,
        )

        # ── Engine A: InceptionTime CNN ensemble ─────────────────────────────
        logger.log("    [Engine A] InceptionTime CNN ensemble ...")
        s2_cnn_weighted_sum = np.zeros(n_routed, dtype=np.float64)
        s2_cnn_f1_total     = 0.0

        for fold_idx, best_f1, ckpt_path, gbdt_model, _ in s2_fold_results:
            fold_num = fold_idx + 1
            if best_f1 <= 0.0 or not os.path.isfile(ckpt_path):
                logger.log(f"      S2 Fold {fold_num}: checkpoint missing / F1=0 – skipping CNN.")
                continue
            logger.log(f"      S2 Fold {fold_num}: loading {ckpt_path}  (Val F1={best_f1:.6f})")

            model_s2 = InceptionTime1D(
                in_channels = cfg["in_channels"],
                num_classes = 1,
                nb_filters  = cfg["nb_filters"],
                bottleneck  = cfg["bottleneck"],
                depth       = cfg["depth"],
            ).to(device)
            model_s2.load_state_dict(torch.load(ckpt_path, map_location=device))
            model_s2.eval()

            fold_probs_s2 = []
            for (x_batch,) in routed_loader:
                x = x_batch.to(device, non_blocking=True)
                # TTA: sigmoid probabilities (probability of being L2)
                p_orig = torch.sigmoid(model_s2(x).squeeze(-1))
                p_up   = torch.sigmoid(model_s2(x * 1.05).squeeze(-1))
                p_down = torch.sigmoid(model_s2(x * 0.95).squeeze(-1))
                p_tta  = (p_orig + p_up + p_down) / 3.0
                fold_probs_s2.append(p_tta.cpu().numpy())

            fold_probs_s2_arr      = np.concatenate(fold_probs_s2, axis=0)   # (n_routed,)
            s2_cnn_weighted_sum   += best_f1 * fold_probs_s2_arr
            s2_cnn_f1_total       += best_f1
            logger.log(f"        → accumulated (weight={best_f1:.6f})")

        if s2_cnn_f1_total > 0.0:
            p_cnn = s2_cnn_weighted_sum / s2_cnn_f1_total   # (n_routed,) prob of L2
        else:
            logger.log("    WARNING: No valid S2 CNN folds – P_CNN defaults to 0.5.")
            p_cnn = np.full(n_routed, 0.5, dtype=np.float64)

        # ── Engine B: GBDT tabular ensemble ─────────────────────────────────
        logger.log("    [Engine B] GBDT tabular feature ensemble ...")
        X_routed_tab = extract_tabular_features(routed_signals_np)   # (n_routed, 23)

        s2_gbdt_weighted_sum = np.zeros(n_routed, dtype=np.float64)
        s2_gbdt_f1_total     = 0.0

        for fold_idx, best_f1, ckpt_path, gbdt_model, _ in s2_fold_results:
            fold_num = fold_idx + 1
            if gbdt_model is None or best_f1 <= 0.0:
                logger.log(f"      S2 Fold {fold_num}: GBDT model not available – skipping.")
                continue
            logger.log(f"      S2 GBDT Fold {fold_num}: predicting  (Val F1={best_f1:.6f})")

            fold_gbdt_probs = gbdt_model.predict_proba(X_routed_tab)[:, 1]  # P(L2)
            s2_gbdt_weighted_sum += best_f1 * fold_gbdt_probs
            s2_gbdt_f1_total     += best_f1
            logger.log(f"        → accumulated (weight={best_f1:.6f})")

        if s2_gbdt_f1_total > 0.0:
            p_gbdt = s2_gbdt_weighted_sum / s2_gbdt_f1_total   # (n_routed,) prob of L2
        else:
            logger.log("    WARNING: No valid S2 GBDT folds – P_GBDT defaults to 0.5.")
            p_gbdt = np.full(n_routed, 0.5, dtype=np.float64)

        # ── Weighted Decision Head ───────────────────────────────────────────
        # P_specialist_final = cnn_weight × P_CNN + gbdt_weight × P_GBDT
        cnn_w  = cfg["cnn_weight"]
        gbdt_w = cfg["gbdt_weight"]
        p_final = cnn_w * p_cnn + gbdt_w * p_gbdt

        logger.log(f"\n    Ensemble weights: CNN={cnn_w:.2f}  GBDT={gbdt_w:.2f}")
        logger.log(f"    Decision threshold: {cfg['s2_threshold']}")

        s2_preds_binary = (p_final >= cfg["s2_threshold"]).astype(np.int64)
        s2_preds_orig   = np.where(s2_preds_binary == 0, 1, 2)  # 0→L1, 1→L2

        # Write Stage 2 decisions into final_preds at the routed positions
        routed_indices = np.where(s2_route_mask)[0]
        for i, idx in enumerate(routed_indices):
            final_preds[idx] = int(s2_preds_orig[i])

        s2_l1_count = int((s2_preds_orig == 1).sum())
        s2_l2_count = int((s2_preds_orig == 2).sum())
        logger.log(f"\n  Stage 2 decisions: L1={s2_l1_count}  L2={s2_l2_count}  "
                   f"(threshold={cfg['s2_threshold']})")
    else:
        logger.log("  No samples routed to Stage 2 – Stage 2 skipped.")

    # Sanity check: all predictions should be assigned
    unassigned = int((final_preds == -1).sum())
    if unassigned > 0:
        logger.log(f"  WARNING: {unassigned} samples have unassigned predictions! Defaulting to L0.")
        final_preds[final_preds == -1] = 0

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Assemble and save submission
    # ══════════════════════════════════════════════════════════════════════════
    submission = (
        pd.DataFrame({"Id": file_ids, "Label": final_preds})
        .sort_values("Id")
        .reset_index(drop=True)
    )
    submission.to_csv(cfg["submission_path"], index=False)

    logger.log(f"\n  ── Submission Summary ──────────────────────────────────────────")
    logger.log(f"  S1 folds used    : {len(s1_fold_results)}")
    logger.log(f"  S2 folds used    : {len(s2_fold_results)}")
    logger.log(f"  Total test       : {n_test}")
    logger.log(f"  Submission path  : {cfg['submission_path']}")
    pred_dist = submission["Label"].value_counts().sort_index()
    logger.log("  Final predicted label distribution:")
    for lbl, cnt in pred_dist.items():
        logger.log(f"    Label {lbl}: {cnt:>5d}  ({cnt / n_test * 100:.1f}%)")

    return submission


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HAR InceptionTime v9 – Hierarchical Two-Stage + GBDT Ensemble")
    parser.add_argument("--all-folds",  action="store_true",
                        help="Run all 5 folds (default: fold 1 only)")
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--batch-size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--patience",   type=int,   default=CFG["patience"])
    parser.add_argument("--threshold",  type=float, default=CFG["s2_threshold"],
                        help="Stage 2 binary decision threshold (default=0.5)")
    args = parser.parse_args()

    CFG["run_all_folds"] = args.all_folds
    CFG["epochs"]        = args.epochs
    CFG["batch_size"]    = args.batch_size
    CFG["lr"]            = args.lr
    CFG["patience"]      = args.patience
    CFG["T_max"]         = args.epochs
    CFG["s2_threshold"]  = args.threshold

    seed_everything(CFG["seed"])

    logger = DualLogger(CFG["log_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gbdt_backend = "LightGBM" if LGBM_AVAILABLE else "XGBoost"

    # ── Header ────────────────────────────────────────────────────────────────
    header_lines = [
        "=" * 70,
        "  HAR InceptionTime v9 – Heterogeneous Dual-Engine + TTFN",
        "=" * 70,
        f"  Log file      : {os.path.abspath(CFG['log_path'])}",
        f"  Device        : {device}",
    ]
    if device.type == "cuda":
        header_lines += [
            f"  GPU           : {torch.cuda.get_device_name(0)}",
            f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        ]
    header_lines += [
        f"  Seed          : {CFG['seed']}  (deterministic=True, benchmark=False)",
        f"  In channels   : {CFG['in_channels']}  (8 stable: mean_x/y/z + std_x/y/z + mean_mag + std_mag)",
        f"  nb_filters    : {CFG['nb_filters']}  (per branch → hidden_dim={CFG['nb_filters']*4})",
        f"  Bottleneck    : {CFG['bottleneck']}",
        f"  Depth         : {CFG['depth']}  (Inception blocks; residual every 3)",
        f"  S1 classes    : {CFG['num_classes_s1']}  (L0 | L1_2_Merged | L3 | L4 | L5)",
        f"  S2 Engine A   : InceptionTime1D (1 logit / BCE)",
        f"  S2 Engine B   : {gbdt_backend} (23 tabular features from std_mag + all-ch stats)",
        f"  S2 Ensemble   : P_final = {CFG['cnn_weight']}×P_CNN + {CFG['gbdt_weight']}×P_GBDT",
        f"  Epochs        : {CFG['epochs']}",
        f"  Batch         : {CFG['batch_size']}",
        f"  LR            : {CFG['lr']}",
        f"  Weight Decay  : {CFG['weight_decay']}",
        f"  Focal γ       : {CFG['focal_gamma']}  (Stage 1 uniform)",
        f"  Label Smooth  : ε={CFG['label_smoothing']}  (Stage 1)",
        f"  Mixup α       : {CFG['mixup_alpha']}  (Stage 1 only)",
        f"  Patience      : {CFG['patience']}",
        f"  S2 Threshold  : {CFG['s2_threshold']}",
        f"  Folds         : {'all 5' if CFG['run_all_folds'] else 'fold 1 only'}",
        f"  S1 Ensemble   : F1-Weighted TTA (×1.00 / ×1.05 / ×0.95)",
        f"  Domain Adapt  : Test-Time Feature Normalisation (intrinsic Z-score)",
        f"  Submission    : {CFG['submission_path']}",
        "=" * 70,
    ]
    for line in header_lines:
        logger.log(line)

    folds_to_run = list(range(N_FOLDS)) if CFG["run_all_folds"] else [0]

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A – Train Stage 1 Generalist
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE A – Stage 1 Generalist Training (5-class InceptionTime)")
    logger.log("═" * 70)

    t_phaseA_start  = time.time()
    s1_fold_results = []   # (fold_idx, best_f1, norm_params, ckpt_path, report)
    s1_all_preds    = []
    s1_all_labels   = []

    for fold_idx in folds_to_run:
        best_f1, norm_params, ckpt_path, report, (val_preds, val_labels) = train_fold_s1(
            fold_idx, device, CFG, logger)
        s1_fold_results.append((fold_idx, best_f1, norm_params, ckpt_path, report))
        if val_preds is not None:
            s1_all_preds.extend(val_preds.tolist())
            s1_all_labels.extend(val_labels.tolist())

    t_phaseA = time.time() - t_phaseA_start

    logger.log("\n" + "═" * 70)
    logger.log("  PHASE A SUMMARY – Stage 1 Cross-Validation")
    logger.log("═" * 70)
    s1_f1_scores = [r[1] for r in s1_fold_results]
    for fi, bf1, _, _, _ in s1_fold_results:
        logger.log(f"  [S1] Fold {fi+1}:  Best Val Macro F1 = {bf1:.6f}")
    logger.log(f"\n  [S1] Mean F1 = {np.mean(s1_f1_scores):.6f}  ±  {np.std(s1_f1_scores):.6f}")
    logger.log(f"  Phase A wall-clock: {t_phaseA / 60:.1f} min")

    if len(s1_all_labels) > 0:
        s1_label_names = ["L0", "L1_2_Merged", "L3", "L4", "L5"]
        cm_text = format_confusion_matrix(s1_all_labels, s1_all_preds, s1_label_names)
        logger.log_file("\n--- [S1] Aggregate Confusion Matrix ---")
        logger.log_file(cm_text)
        logger.log("\n--- [S1] Aggregate Confusion Matrix ---")
        logger.log(cm_text)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE B – Train Stage 2 Specialist (InceptionTime + GBDT)
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE B – Stage 2 Specialist Training "
               "(L1 vs L2 | InceptionTime + GBDT Dual-Engine)")
    logger.log("═" * 70)

    t_phaseB_start  = time.time()
    s2_fold_results = []   # (fold_idx, best_f1, ckpt_path, gbdt_model, report)

    for fold_idx in folds_to_run:
        best_f1, ckpt_path, gbdt_model, report = train_fold_s2(
            fold_idx, device, CFG, logger)
        s2_fold_results.append((fold_idx, best_f1, ckpt_path, gbdt_model, report))

    t_phaseB = time.time() - t_phaseB_start

    logger.log("\n" + "═" * 70)
    logger.log("  PHASE B SUMMARY – Stage 2 Cross-Validation")
    logger.log("═" * 70)
    s2_f1_scores = [r[1] for r in s2_fold_results]
    for fi, bf1, _, _, _ in s2_fold_results:
        logger.log(f"  [S2] Fold {fi+1}:  Best Val Macro F1 = {bf1:.6f}")
    if len(s2_f1_scores) > 0:
        logger.log(f"\n  [S2] Mean F1 = {np.mean(s2_f1_scores):.6f}  ±  {np.std(s2_f1_scores):.6f}")
    logger.log(f"  Phase B wall-clock: {t_phaseB / 60:.1f} min")
    logger.log(f"  Total A+B wall-clock: {(t_phaseA + t_phaseB) / 60:.1f} min")

    # ════════════════════════════════════════════════════════════════════════
    # INFERENCE – Hierarchical Routing with TTFN + Dual-Engine
    # ════════════════════════════════════════════════════════════════════════
    if len(s1_fold_results) > 0:
        submission = hierarchical_inference(
            s1_fold_results=s1_fold_results,
            s2_fold_results=s2_fold_results,
            cfg=CFG,
            device=device,
            logger=logger,
        )

    logger.log("\n" + "=" * 70)
    logger.log("  Pipeline v9 (InceptionTime + GBDT Dual-Engine + TTFN) complete.")
    logger.log(f"  Hierarchical submission: {CFG['submission_path']}")
    logger.log("=" * 70)

    logger.close()


if __name__ == "__main__":
    main()
