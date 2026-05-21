"""
train.py
--------
Flat 6-Class SE-InceptionTime + LightGBM Heterogeneous Ensemble Pipeline (v10).

Architecture
────────────
  Global Flat 6-Class Task  : L0, L1, L2, L3, L4, L5
  Engine A (DL)  : SE-InceptionTime1D – 6-class flat classifier
  Engine B (GBDT): LightGBM on 100+ handcrafted time-frequency tabular features

Training Phases
───────────────
  Phase A – SE-InceptionTime (Engine A):
    • 5-fold StratifiedGroupKFold on flat 6-class labels
    • Focal Loss (γ=2.0, ε=0.1 label smoothing, α=inverse-frequency)
    • Batch-level Mixup (Beta(α=0.2))
    • AdamW + CosineAnnealingLR
    • Early stop on validation Macro F1 (patience=25)
    • Saves: checkpoints/v10_fold_{n}_best.pth
    • Accumulates full (N, 6) OOF probability matrices

  Phase B – LightGBM (Engine B):
    • Same 5-fold splits as Phase A
    • 100+ time-frequency features per 8-channel signal window
    • Global 6-class LightGBM classifier per fold
    • Accumulates full (N, 6) OOF probability matrices

OOF Blending Vector Optimization
─────────────────────────────────
  • Collects full OOF matrices from both engines across all folds
  • scipy.optimize.minimize searches for W=[w0..w5] ∈ [0,1]^6
  • P_blend[c] = w_c × P_CNN[c] + (1-w_c) × P_GBDT[c]
  • Objective: maximize global validation Macro F1

Inference
─────────
  1. Test-Time Feature Normalization (TTFN) on test signals
  2. SE-InceptionTime ensemble (F1-weighted TTA ×3 scales)
  3. LightGBM ensemble on tabular features
  4. Class-wise blend with optimized W vector
  5. Argmax → submission_v10_ensemble.csv

Usage
─────
    conda activate PyTorch
    python train.py               # fold 1 only
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
from scipy.optimize import minimize
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
    raise ImportError(
        "LightGBM is not installed. Please install: pip install lightgbm")

from dataset import (
    build_fold_datasets, build_test_dataset,
    load_all_samples, load_test_samples, get_fold_splits,
    normalize, HARDataset,
    N_FOLDS, N_CLASSES,
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

NUM_CLASSES = 6   # flat 6-class: L0, L1, L2, L3, L4, L5

CFG = {
    # Paths ───────────────────────────────────────────────────────────────────
    "train_root"        : "./train/train",
    "test_root"         : "./test/test",
    "submission_path"   : "./submission_v10_ensemble.csv",
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

    # Model (SE-InceptionTime) ────────────────────────────────────────────────
    "in_channels"       : 8,       # 8 stable channels
    "nb_filters"        : 64,      # output channels per branch per block (v10 upgraded)
    "bottleneck"        : 32,      # bottleneck projection dim
    "depth"             : 6,       # number of stacked SE-Inception blocks
    "se_reduction"      : 16,      # SE channel reduction ratio r

    # Flat 6-class
    "num_classes"       : NUM_CLASSES,

    # Focal Loss
    "focal_gamma"       : 2.0,
    "label_smoothing"   : 0.1,

    # Mixup
    "mixup_alpha"       : 0.2,

    # Cross-validation ────────────────────────────────────────────────────────
    "run_all_folds"     : False,

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
# Module 2: Focal Loss (6-class flat)
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
# Module 3: Mixup augmentation
# ══════════════════════════════════════════════════════════════════════════════

def mixup_batch(signals:         torch.Tensor,
                labels:          torch.Tensor,
                alpha:           float,
                num_classes:     int,
                label_smoothing: float,
                device:          torch.device):
    """
    Batch-level Mixup with label smoothing.
    Returns x_mixed (B, C, L), y_mixed (B, num_classes), lam.
    """
    B   = signals.size(0)
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
# Module 4: Comprehensive Time-Frequency Feature Extractor (100+ features)
# ══════════════════════════════════════════════════════════════════════════════

def _zero_crossing_rate(signal_1d: np.ndarray) -> float:
    """Zero-crossing rate for a 1-D signal."""
    signs = np.sign(signal_1d)
    # Replace zero-signs with previous sign to avoid ambiguity
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings) / max(len(signal_1d) - 1, 1)


def extract_tabular_features(signals_np: np.ndarray) -> np.ndarray:
    """
    Extract 100+ handcrafted time-frequency features from (N, 8, 300) signal windows.

    Feature layout per channel (17 features × 8 channels = 136 total):
    ─────────────────────────────────────────────────────────────────────
    Time-Domain Statistics (10 features):
        mean, std, median, max, min, variance, skewness, kurtosis,
        25th percentile (Q1), 75th percentile (Q3)

    Time-Domain Dynamics from first derivative X' = X[t] - X[t-1] (4 features):
        derivative_mean, derivative_std,
        derivative_zero_crossing_rate, derivative_energy

    Frequency-Domain (FFT) Spectra (3 features):
        fft_dc_component   (magnitude of 0Hz / DC term)
        fft_max_amplitude  (maximum spectral amplitude across all bins)
        fft_spectral_energy (sum of squared magnitudes of all FFT coefficients)

    Total: 17 × 8 = 136 features per sample.

    Parameters
    ----------
    signals_np : np.ndarray, shape (N, 8, 300)

    Returns
    -------
    features : np.ndarray, shape (N, 136)
    """
    N = signals_np.shape[0]
    feature_rows = []

    for i in range(N):
        sample = signals_np[i]   # (8, 300)
        row_feats = []

        for ch in range(8):
            sig = sample[ch].astype(np.float64)   # (300,)

            # ── Time-Domain Statistics (10 features) ──────────────────────────
            feat_mean     = float(np.mean(sig))
            feat_std      = float(np.std(sig))
            feat_median   = float(np.median(sig))
            feat_max      = float(np.max(sig))
            feat_min      = float(np.min(sig))
            feat_var      = float(np.var(sig))
            feat_skew     = float(scipy_skew(sig, bias=True))
            feat_kurt     = float(scipy_kurtosis(sig, fisher=True, bias=True))
            feat_q25      = float(np.percentile(sig, 25))
            feat_q75      = float(np.percentile(sig, 75))

            # ── Time-Domain Dynamics: 1st derivative (4 features) ─────────────
            deriv = np.diff(sig)                              # (299,)
            deriv_mean   = float(np.mean(deriv))
            deriv_std    = float(np.std(deriv))
            deriv_zcr    = _zero_crossing_rate(deriv)
            deriv_energy = float(np.sum(deriv ** 2))

            # ── Frequency-Domain: FFT (3 features) ────────────────────────────
            fft_coeffs  = np.fft.rfft(sig)                   # (151,) complex
            fft_mag     = np.abs(fft_coeffs)                  # magnitude spectrum
            fft_dc      = float(fft_mag[0])                   # DC component (0 Hz)
            fft_max_amp = float(np.max(fft_mag))              # max spectral amplitude
            fft_energy  = float(np.sum(fft_mag ** 2))         # total spectral energy

            # Collect all 17 features for this channel
            ch_feats = [
                feat_mean, feat_std, feat_median, feat_max, feat_min,
                feat_var, feat_skew, feat_kurt, feat_q25, feat_q75,
                deriv_mean, deriv_std, deriv_zcr, deriv_energy,
                fft_dc, fft_max_amp, fft_energy,
            ]
            row_feats.extend(ch_feats)

        feature_rows.append(np.array(row_feats, dtype=np.float32))

    return np.stack(feature_rows, axis=0)   # (N, 136)


N_TAB_FEATURES = 136   # 17 features × 8 channels


# ══════════════════════════════════════════════════════════════════════════════
# Module 5: Global LightGBM Engine (6-class)
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm_model() -> lgb.LGBMClassifier:
    """Instantiate a global 6-class LightGBM classifier with optimised defaults."""
    return lgb.LGBMClassifier(
        objective        = "multiclass",
        num_class        = NUM_CLASSES,
        n_estimators     = 800,
        learning_rate    = 0.05,
        max_depth        = 7,
        num_leaves       = 63,
        min_child_samples= 20,
        subsample        = 0.8,
        subsample_freq   = 1,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        class_weight     = "balanced",
        random_state     = 42,
        verbose          = -1,
        n_jobs           = -1,
    )


def _collect_raw_signals_from_loader(loader) -> tuple:
    """
    Iterate a DataLoader and return concatenated raw signal arrays.
    Returns (signals_np (N,8,300), labels_np (N,)).
    """
    all_signals, all_labels = [], []
    for signals, labels in loader:
        all_signals.append(signals.numpy())
        all_labels.append(labels.numpy())
    return (np.concatenate(all_signals, axis=0),
            np.concatenate(all_labels,  axis=0))


# ══════════════════════════════════════════════════════════════════════════════
# DL training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device,
                    mixup_alpha: float, label_smoothing: float,
                    num_classes: int) -> float:
    """Train SE-InceptionTime for one epoch with Mixup + Focal Loss."""
    model.train()
    running_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

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
def evaluate(model, loader, criterion, device):
    """Validation pass: returns (avg_loss, accuracy, macro_f1, preds, labels, probs)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        logits  = model(signals)
        running_loss += criterion(logits, labels).item() * signals.size(0)
        probs   = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)   # (N, 6)

    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels, all_probs


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix formatting helper
# ══════════════════════════════════════════════════════════════════════════════

def format_confusion_matrix(all_labels, all_preds, label_names=None) -> str:
    labels  = sorted(list(set(np.concatenate([np.unique(all_labels),
                                               np.unique(all_preds)]))))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    if label_names is None:
        label_names = [str(i) for i in labels]
    header  = "Pred→  " + "  ".join(f"  {ln[:4]:>4}" for ln in label_names)
    divider = "─" * len(header)
    rows    = [header, divider]
    for i, row in enumerate(cm):
        rows.append(f"True {label_names[i][:4]:>4} | " +
                    " | ".join(f"{v:5d}" for v in row))
    rows.append("(Rows = True Label, Columns = Predicted Label)")
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Per-fold SE-InceptionTime training (flat 6-class)
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["L0", "L1", "L2", "L3", "L4", "L5"]


def train_fold_dl(fold_idx: int, device: torch.device, cfg: dict,
                  logger: DualLogger):
    """
    Train one CV fold for the flat 6-class SE-InceptionTime.

    Returns
    -------
    best_val_f1     : float
    norm_params     : (mean, std)
    checkpoint_path : str
    best_val_report : str
    oof_probs       : np.ndarray (N_val, 6)  – OOF probability matrix
    oof_labels      : np.ndarray (N_val,)    – true labels for this fold
    val_indices     : np.ndarray (N_val,)    – absolute sample indices
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase A] --- Fold {fold_num}/{N_FOLDS} SE-InceptionTime Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE A | FOLD {fold_num} / {N_FOLDS}  "
                    f"– Flat 6-Class SE-InceptionTime")
    logger.log_file("=" * 70)

    train_ds, val_ds, norm_params, class_counts, n_cls = build_fold_datasets(
        cfg["train_root"], fold_idx=fold_idx, mode="flat"
    )

    logger.log_file(f"  Training samples  : {len(train_ds)}")
    logger.log_file(f"  Validation samples: {len(val_ds)}")
    logger.log_file("  Class counts (train fold):")
    for c, cnt in enumerate(class_counts):
        logger.log_file(f"    {CLASS_NAMES[c]}: {cnt:>5d}  "
                        f"({cnt / max(class_counts.sum(), 1) * 100:.1f}%)")
    logger.log_console(f"  [DL] Train: {len(train_ds)}  Val: {len(val_ds)}  "
                       f"Classes: {class_counts.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    model = InceptionTime1D(
        in_channels  = cfg["in_channels"],
        num_classes  = cfg["num_classes"],
        nb_filters   = cfg["nb_filters"],
        bottleneck   = cfg["bottleneck"],
        depth        = cfg["depth"],
        se_reduction = cfg["se_reduction"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_file(f"  Model: SE-InceptionTime1D (nb_filters={cfg['nb_filters']}, "
                    f"depth={cfg['depth']}, SE r={cfg['se_reduction']})  "
                    f"|  Params: {n_params:,}")

    criterion = build_focal_loss(
        class_counts, cfg["num_classes"], device,
        gamma=cfg["focal_gamma"],
        label_smoothing=cfg["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path  = os.path.join(cfg["checkpoint_dir"],
                              f"v10_fold_{fold_num}_best.pth")
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    best_val_preds  = (None, None)
    best_val_probs  = None
    best_val_report = ""

    for epoch in range(1, cfg["epochs"] + 1):
        t_start    = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=cfg["mixup_alpha"],
            label_smoothing=cfg["label_smoothing"],
            num_classes=cfg["num_classes"],
        )
        scheduler.step()
        val_loss, val_acc, val_f1, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - t_start
        lr_now  = scheduler.get_last_lr()[0]

        logger.log_console(
            f"[DL|Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.1e} | "
            f"Loss: T:{train_loss:.2f} / V:{val_loss:.2f} | "
            f"Val Acc: {val_acc:5.2f}% | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        logger.log_file(
            f"[DL|Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.4e} | "
            f"Loss: T:{train_loss:.6f} / V:{val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f}% | "
            f"Val F1: {val_f1:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        if val_f1 >= early_stop.best_score:
            best_val_preds  = (val_preds, val_labels)
            best_val_probs  = val_probs.copy()
            best_val_report = classification_report(
                val_labels, val_preds,
                target_names=CLASS_NAMES, digits=4, zero_division=0)

        should_stop = early_stop.step(val_f1, model, epoch)
        if should_stop:
            msg = (f"  >> [DL|Fold {fold_num}] Early stop @ epoch {epoch}. "
                   f"Best epoch: {early_stop.best_epoch}  "
                   f"Best F1: {early_stop.best_score:.4f}")
            logger.log(msg)
            break

    fold_summary = (
        f"--- [DL] Fold {fold_num} done | "
        f"Best Val Macro F1: {early_stop.best_score:.6f} "
        f"(epoch {early_stop.best_epoch}) ---"
    )
    logger.log(fold_summary)
    logger.log_file("")
    logger.log_file(f"  [DL] Classification Report @ Best Epoch ({early_stop.best_epoch}):")
    logger.log_file(best_val_report)
    logger.log_file("─" * 70)

    oof_labels = best_val_preds[1] if best_val_preds[1] is not None else np.array([])
    oof_probs  = best_val_probs   if best_val_probs   is not None else np.zeros((0, NUM_CLASSES))

    return (early_stop.best_score, norm_params, ckpt_path,
            best_val_report, oof_probs, oof_labels)


# ══════════════════════════════════════════════════════════════════════════════
# Phase B: Per-fold LightGBM training (flat 6-class)
# ══════════════════════════════════════════════════════════════════════════════

def train_fold_gbdt(fold_idx: int, cfg: dict, logger: DualLogger):
    """
    Train one CV fold for the flat 6-class LightGBM engine.

    Returns
    -------
    best_val_f1 : float
    gbdt_model  : fitted LGBMClassifier
    oof_probs   : np.ndarray (N_val, 6)  – OOF probability matrix
    oof_labels  : np.ndarray (N_val,)    – true labels
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase B] --- Fold {fold_num}/{N_FOLDS} LightGBM Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE B | FOLD {fold_num} / {N_FOLDS}  "
                    f"– Flat 6-Class LightGBM (100+ tabular features)")
    logger.log_file("=" * 70)

    # Reuse the same fold splits (flat mode)
    train_ds, val_ds, _, class_counts, _ = build_fold_datasets(
        cfg["train_root"], fold_idx=fold_idx, mode="flat"
    )

    # Collect raw signals (not normalised) for tabular feature extraction
    train_loader_ord = DataLoader(train_ds, batch_size=512,
                                  shuffle=False, num_workers=cfg["num_workers"],
                                  pin_memory=False)
    val_loader_ord   = DataLoader(val_ds,   batch_size=512,
                                  shuffle=False, num_workers=cfg["num_workers"],
                                  pin_memory=False)

    train_signals_np, train_labels_np = _collect_raw_signals_from_loader(train_loader_ord)
    val_signals_np,   val_labels_np   = _collect_raw_signals_from_loader(val_loader_ord)

    logger.log(f"  [GBDT|Fold {fold_num}] Extracting tabular features "
               f"({N_TAB_FEATURES} features per sample) ...")

    t_feat = time.time()
    X_train = extract_tabular_features(train_signals_np)   # (N_train, 136)
    X_val   = extract_tabular_features(val_signals_np)     # (N_val,   136)
    feat_elapsed = time.time() - t_feat

    logger.log(f"  [GBDT|Fold {fold_num}] Feature extraction done: "
               f"train={X_train.shape}, val={X_val.shape}  "
               f"Time: {feat_elapsed:.1f}s")
    logger.log_file(f"  Feature matrix: train={X_train.shape}  val={X_val.shape}")

    gbdt_model = build_lgbm_model()

    t_gbdt = time.time()
    gbdt_model.fit(X_train, train_labels_np)
    gbdt_elapsed = time.time() - t_gbdt

    # OOF predictions
    oof_probs  = gbdt_model.predict_proba(X_val)           # (N_val, 6)
    oof_preds  = oof_probs.argmax(axis=1)
    gbdt_val_f1  = f1_score(val_labels_np, oof_preds, average="macro", zero_division=0)
    gbdt_val_acc = accuracy_score(val_labels_np, oof_preds) * 100.0

    logger.log(f"  [GBDT|Fold {fold_num}] LightGBM Val Macro F1: {gbdt_val_f1:.4f}  "
               f"Acc: {gbdt_val_acc:.2f}%  Train time: {gbdt_elapsed:.1f}s")
    logger.log_file(f"  [GBDT|Fold {fold_num}] LightGBM Val Macro F1: {gbdt_val_f1:.6f}  "
                    f"Acc: {gbdt_val_acc:.4f}%  Train time: {gbdt_elapsed:.2f}s")
    logger.log_file(f"  [GBDT|Fold {fold_num}] Classification Report:")
    logger.log_file(classification_report(
        val_labels_np, oof_preds, target_names=CLASS_NAMES,
        digits=4, zero_division=0))
    logger.log_file("─" * 70)

    return gbdt_val_f1, gbdt_model, oof_probs, val_labels_np


# ══════════════════════════════════════════════════════════════════════════════
# Module 6: 6-class OOF Blending Vector Optimization
# ══════════════════════════════════════════════════════════════════════════════

def optimize_blend_weights(oof_probs_dl:   np.ndarray,
                           oof_probs_gbdt: np.ndarray,
                           oof_labels:     np.ndarray,
                           logger:         DualLogger) -> np.ndarray:
    """
    Search for an optimal 6-dimensional blending weight vector W ∈ [0,1]^6
    that maximizes the global validation Macro F1 score.

    Blending rule:
        P_blend[c] = w_c × P_CNN[c] + (1 - w_c) × P_GBDT[c]

    Uses scipy.optimize.minimize (L-BFGS-B) with random multi-start restarts
    to escape local optima.

    Parameters
    ----------
    oof_probs_dl   : (N_val_total, 6) – accumulated DL OOF probs
    oof_probs_gbdt : (N_val_total, 6) – accumulated GBDT OOF probs
    oof_labels     : (N_val_total,)   – true integer labels

    Returns
    -------
    best_w : np.ndarray, shape (6,)
             Optimal weight vector W where w_c ∈ [0, 1]
    """
    logger.log("\n" + "=" * 70)
    logger.log("  OOF BLENDING WEIGHT OPTIMIZATION  (6-dimensional search)")
    logger.log("=" * 70)
    logger.log(f"  OOF samples: {len(oof_labels)}  "
               f"DL shape: {oof_probs_dl.shape}  "
               f"GBDT shape: {oof_probs_gbdt.shape}")

    # Baseline: pure DL
    preds_dl   = oof_probs_dl.argmax(axis=1)
    f1_dl      = f1_score(oof_labels, preds_dl, average="macro", zero_division=0)
    # Baseline: pure GBDT
    preds_gbdt = oof_probs_gbdt.argmax(axis=1)
    f1_gbdt    = f1_score(oof_labels, preds_gbdt, average="macro", zero_division=0)
    # Equal blend
    blend_eq   = 0.5 * oof_probs_dl + 0.5 * oof_probs_gbdt
    f1_eq      = f1_score(oof_labels, blend_eq.argmax(axis=1),
                          average="macro", zero_division=0)

    logger.log(f"  Baseline — Pure DL  F1: {f1_dl:.6f}")
    logger.log(f"  Baseline — Pure GBDT F1: {f1_gbdt:.6f}")
    logger.log(f"  Baseline — Equal Blend (0.5) F1: {f1_eq:.6f}")

    def neg_macro_f1(w: np.ndarray) -> float:
        """Objective: negative macro F1 (minimized by scipy)."""
        w_clipped = np.clip(w, 0.0, 1.0)
        # Class-wise blend: P_blend[:, c] = w[c]*P_dl[:, c] + (1-w[c])*P_gbdt[:, c]
        blend = w_clipped[np.newaxis, :] * oof_probs_dl + \
                (1.0 - w_clipped)[np.newaxis, :] * oof_probs_gbdt
        preds = blend.argmax(axis=1)
        return -f1_score(oof_labels, preds, average="macro", zero_division=0)

    bounds = [(0.0, 1.0)] * NUM_CLASSES

    best_result = None
    best_f1     = -np.inf

    # Multi-start random restarts (20 starts for thorough search)
    rng = np.random.RandomState(42)
    n_starts = 20
    logger.log(f"\n  Running L-BFGS-B optimization with {n_starts} random restarts ...")

    for start_idx in range(n_starts):
        if start_idx == 0:
            w0 = np.full(NUM_CLASSES, 0.5)    # start from equal blend
        elif start_idx == 1:
            w0 = np.ones(NUM_CLASSES) * 0.8   # DL-heavy start
        elif start_idx == 2:
            w0 = np.ones(NUM_CLASSES) * 0.2   # GBDT-heavy start
        else:
            w0 = rng.uniform(0.0, 1.0, NUM_CLASSES)

        try:
            result = minimize(neg_macro_f1, w0, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
            candidate_f1 = -result.fun
            if candidate_f1 > best_f1:
                best_f1     = candidate_f1
                best_result = result
        except Exception:
            continue

    if best_result is None:
        logger.log("  WARNING: Optimizer failed all starts. Falling back to equal blend.")
        best_w = np.full(NUM_CLASSES, 0.5)
        best_f1 = f1_eq
    else:
        best_w = np.clip(best_result.x, 0.0, 1.0)

    logger.log(f"\n  Optimized Blending Weight Vector W:")
    for c in range(NUM_CLASSES):
        logger.log(f"    w_{c} ({CLASS_NAMES[c]}): {best_w[c]:.6f}  "
                   f"→ P_blend = {best_w[c]:.4f}×P_CNN + "
                   f"{1-best_w[c]:.4f}×P_GBDT")
    logger.log(f"\n  OOF Macro F1 with optimized W: {best_f1:.6f}")
    logger.log(f"  Improvement over pure DL  : {best_f1 - f1_dl:+.6f}")
    logger.log(f"  Improvement over pure GBDT: {best_f1 - f1_gbdt:+.6f}")
    logger.log(f"  Improvement over 0.5 blend: {best_f1 - f1_eq:+.6f}")

    return best_w


# ══════════════════════════════════════════════════════════════════════════════
# Module 7: Flat 6-Class Blending Inference with TTFN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def flat_ensemble_inference(dl_fold_results:   list,
                             gbdt_fold_results: list,
                             blend_weights:     np.ndarray,
                             cfg:               dict,
                             device:            torch.device,
                             logger:            DualLogger) -> pd.DataFrame:
    """
    Flat 6-class blending inference pipeline (v10).

    Step 1: Test-Time Feature Normalization (TTFN) – intrinsic Z-score on test batch
    Step 2: SE-InceptionTime ensemble (F1-weighted TTA ×3 scales) → (N, 6) probs
    Step 3: LightGBM ensemble on tabular features → (N, 6) probs
    Step 4: Class-wise blend via optimized W vector
    Step 5: Argmax → submission_v10_ensemble.csv

    Parameters
    ----------
    dl_fold_results   : list of (fold_idx, best_f1, norm_params, ckpt_path, report, oof_p, oof_l)
    gbdt_fold_results : list of (fold_idx, best_f1, gbdt_model, oof_p, oof_l)
    blend_weights     : np.ndarray (6,) – optimized class-wise weight vector W
    cfg               : dict
    device            : torch.device
    logger            : DualLogger

    Returns
    -------
    submission : pd.DataFrame  with columns ['Id', 'Label']
    """
    logger.log("\n" + "=" * 70)
    logger.log("  FLAT 6-CLASS BLENDING INFERENCE  (SE-InceptionTime + LightGBM + TTFN)")
    logger.log("=" * 70)

    # ── Test-Time Feature Normalization ──────────────────────────────────────
    logger.log("\n  [TTFN] Loading test data with intrinsic Z-score normalisation ...")

    raw_test_signals_300_8, file_ids = load_test_samples(cfg["test_root"])

    # Transpose to (N, 8, 300)
    raw_test_signals = raw_test_signals_300_8.transpose(0, 2, 1).astype(np.float32)
    n_test = raw_test_signals.shape[0]

    # Compute per-channel mean and std across test batch: (8, N*300)
    test_flat = raw_test_signals.transpose(1, 0, 2).reshape(8, -1)
    test_mean = test_flat.mean(axis=1, keepdims=True)   # (8, 1)
    test_std  = test_flat.std(axis=1,  keepdims=True)   # (8, 1)
    test_std  = np.where(test_std < 1e-8, 1.0, test_std)

    # Intrinsic standardisation: (N, 8, 300)
    normed_test = (raw_test_signals - test_mean[np.newaxis, :, :]) / \
                   test_std[np.newaxis, :, :]
    normed_test = normed_test.astype(np.float32)

    logger.log(f"  [TTFN] N_test={n_test}  "
               f"Per-channel mean: {np.round(test_mean.squeeze(), 4).tolist()}")
    logger.log(f"  [TTFN] Per-channel std:  {np.round(test_std.squeeze(), 4).tolist()}")

    # Build DataLoader from normalised test tensor
    normed_tensor   = torch.from_numpy(normed_test)
    dummy_labels    = torch.zeros(n_test, dtype=torch.long)
    normed_test_ds  = TensorDataset(normed_tensor, dummy_labels)
    normed_test_loader = DataLoader(
        normed_test_ds,
        batch_size  = cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = cfg["pin_memory"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: SE-InceptionTime ensemble (F1-weighted TTA ×3 scales)
    # ══════════════════════════════════════════════════════════════════════════
    logger.log("\n  [Step 2] SE-InceptionTime Ensemble ...")

    dl_weighted_prob_sum = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    dl_f1_total          = 0.0

    for fold_idx, best_f1, norm_params, ckpt_path, _, _, _ in dl_fold_results:
        fold_num = fold_idx + 1
        logger.log(f"    DL Fold {fold_num}: loading {ckpt_path}  "
                   f"(Val F1={best_f1:.6f})")

        model = InceptionTime1D(
            in_channels  = cfg["in_channels"],
            num_classes  = cfg["num_classes"],
            nb_filters   = cfg["nb_filters"],
            bottleneck   = cfg["bottleneck"],
            depth        = cfg["depth"],
            se_reduction = cfg["se_reduction"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        model.eval()

        fold_probs = []
        for signals, _ in normed_test_loader:
            x      = signals.to(device, non_blocking=True)
            # TTA: original / scale-up (+5%) / scale-down (−5%)
            p_orig = F.softmax(model(x),        dim=1)
            p_up   = F.softmax(model(x * 1.05), dim=1)
            p_down = F.softmax(model(x * 0.95), dim=1)
            p_tta  = (p_orig + p_up + p_down) / 3.0
            fold_probs.append(p_tta.cpu().numpy())

        fold_probs_arr        = np.concatenate(fold_probs, axis=0)   # (N, 6)
        dl_weighted_prob_sum += best_f1 * fold_probs_arr
        dl_f1_total          += best_f1
        logger.log(f"      → accumulated (weight={best_f1:.6f})")

    if dl_f1_total > 0.0:
        p_dl = dl_weighted_prob_sum / dl_f1_total   # (N, 6)
    else:
        logger.log("  WARNING: No valid DL folds. P_DL defaults to uniform.")
        p_dl = np.full((n_test, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: LightGBM ensemble on tabular features
    # ══════════════════════════════════════════════════════════════════════════
    logger.log("\n  [Step 3] LightGBM Tabular Feature Ensemble ...")

    # Extract tabular features from raw (un-normalised) test signals
    logger.log(f"    Extracting {N_TAB_FEATURES} tabular features from test set ...")
    t_feat = time.time()
    X_test_tab = extract_tabular_features(raw_test_signals)   # (N, 136)
    logger.log(f"    Feature extraction done: {X_test_tab.shape}  "
               f"Time: {time.time()-t_feat:.1f}s")

    gbdt_weighted_prob_sum = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    gbdt_f1_total          = 0.0

    for fold_idx, best_f1, gbdt_model, _, _ in gbdt_fold_results:
        fold_num = fold_idx + 1
        if gbdt_model is None or best_f1 <= 0.0:
            logger.log(f"    GBDT Fold {fold_num}: model not available – skipping.")
            continue
        logger.log(f"    GBDT Fold {fold_num}: predicting  (Val F1={best_f1:.6f})")

        fold_gbdt_probs        = gbdt_model.predict_proba(X_test_tab)  # (N, 6)
        gbdt_weighted_prob_sum += best_f1 * fold_gbdt_probs
        gbdt_f1_total          += best_f1
        logger.log(f"      → accumulated (weight={best_f1:.6f})")

    if gbdt_f1_total > 0.0:
        p_gbdt = gbdt_weighted_prob_sum / gbdt_f1_total   # (N, 6)
    else:
        logger.log("  WARNING: No valid GBDT folds. P_GBDT defaults to uniform.")
        p_gbdt = np.full((n_test, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Class-wise blend with optimized W
    # ══════════════════════════════════════════════════════════════════════════
    logger.log("\n  [Step 4] Class-wise blending with optimized W ...")
    logger.log(f"    W = {np.round(blend_weights, 6).tolist()}")

    # P_final[c] = w_c × P_CNN[c] + (1-w_c) × P_GBDT[c]
    p_final = (blend_weights[np.newaxis, :] * p_dl +
               (1.0 - blend_weights)[np.newaxis, :] * p_gbdt)   # (N, 6)

    # ── Argmax → final label predictions ─────────────────────────────────────
    final_preds = p_final.argmax(axis=1).astype(np.int64)   # (N,)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Assemble and save submission
    # ══════════════════════════════════════════════════════════════════════════
    submission = (
        pd.DataFrame({"Id": file_ids, "Label": final_preds})
        .sort_values("Id")
        .reset_index(drop=True)
    )
    submission.to_csv(cfg["submission_path"], index=False)

    logger.log(f"\n  ── Submission Summary ──────────────────────────────────────────")
    logger.log(f"  DL  folds used    : {len(dl_fold_results)}")
    logger.log(f"  GBDT folds used   : {len(gbdt_fold_results)}")
    logger.log(f"  Total test samples: {n_test}")
    logger.log(f"  Submission path   : {cfg['submission_path']}")
    pred_dist = submission["Label"].value_counts().sort_index()
    logger.log("  Final predicted label distribution:")
    for lbl, cnt in pred_dist.items():
        lbl_name = CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else str(lbl)
        logger.log(f"    {lbl_name} (Label {lbl}): {cnt:>5d}  "
                   f"({cnt / n_test * 100:.1f}%)")

    return submission


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HAR SE-InceptionTime v10 – Flat 6-Class + GBDT Ensemble + OOF Blend")
    parser.add_argument("--all-folds",  action="store_true",
                        help="Run all 5 folds (default: fold 1 only)")
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--batch-size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--patience",   type=int,   default=CFG["patience"])
    args = parser.parse_args()

    CFG["run_all_folds"] = args.all_folds
    CFG["epochs"]        = args.epochs
    CFG["batch_size"]    = args.batch_size
    CFG["lr"]            = args.lr
    CFG["patience"]      = args.patience
    CFG["T_max"]         = args.epochs

    seed_everything(CFG["seed"])

    logger = DualLogger(CFG["log_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    header_lines = [
        "=" * 70,
        "  HAR SE-InceptionTime v10 – Flat 6-Class Heterogeneous Ensemble + TTFN",
        "=" * 70,
        f"  Log file      : {os.path.abspath(CFG['log_path'])}",
        f"  Device        : {device}",
    ]
    if device.type == "cuda":
        header_lines += [
            f"  GPU           : {torch.cuda.get_device_name(0)}",
            f"  VRAM          : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        ]
    hidden_dim = CFG["nb_filters"] * 4
    header_lines += [
        f"  Seed          : {CFG['seed']}  (deterministic=True, benchmark=False)",
        f"  Architecture  : SE-InceptionTime1D (flat 6-class, SE block per Inception block)",
        f"  In channels   : {CFG['in_channels']}  "
        f"(8 stable: mean_x/y/z + std_x/y/z + mean_mag + std_mag)",
        f"  nb_filters    : {CFG['nb_filters']}  "
        f"(per branch → hidden_dim={hidden_dim})",
        f"  Bottleneck    : {CFG['bottleneck']}",
        f"  Depth         : {CFG['depth']}  (SE-Inception blocks; residual every 3)",
        f"  SE reduction  : r={CFG['se_reduction']}",
        f"  Num classes   : {CFG['num_classes']}  (L0 | L1 | L2 | L3 | L4 | L5 – flat)",
        f"  GBDT Engine   : LightGBM ({N_TAB_FEATURES} tabular features: "
        f"17 per channel × 8 channels)",
        f"  Feature types : Time-stats(10) + Derivative(4) + FFT(3) per channel",
        f"  Blend Optimizer: scipy L-BFGS-B (20 random restarts, 6D weight vector)",
        f"  Epochs        : {CFG['epochs']}",
        f"  Batch         : {CFG['batch_size']}",
        f"  LR            : {CFG['lr']}",
        f"  Weight Decay  : {CFG['weight_decay']}",
        f"  Focal γ       : {CFG['focal_gamma']}",
        f"  Label Smooth  : ε={CFG['label_smoothing']}",
        f"  Mixup α       : {CFG['mixup_alpha']}",
        f"  Patience      : {CFG['patience']}",
        f"  Folds         : {'all 5' if CFG['run_all_folds'] else 'fold 1 only'}",
        f"  DL Ensemble   : F1-Weighted TTA (×1.00 / ×1.05 / ×0.95)",
        f"  Domain Adapt  : Test-Time Feature Normalisation (intrinsic Z-score)",
        f"  Submission    : {CFG['submission_path']}",
        "=" * 70,
    ]
    for line in header_lines:
        logger.log(line)

    folds_to_run = list(range(N_FOLDS)) if CFG["run_all_folds"] else [0]

    # Accumulators for OOF matrices (filled across all folds)
    oof_probs_dl_list   = []   # Each entry: (N_val, 6)
    oof_probs_gbdt_list = []   # Each entry: (N_val, 6)
    oof_labels_list     = []   # Each entry: (N_val,)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A – Train SE-InceptionTime (flat 6-class)
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE A – SE-InceptionTime Training (Flat 6-Class)")
    logger.log("═" * 70)

    t_phaseA_start = time.time()
    dl_fold_results = []
    dl_all_preds    = []
    dl_all_labels   = []

    for fold_idx in folds_to_run:
        (best_f1, norm_params, ckpt_path, report,
         oof_probs, oof_labels) = train_fold_dl(fold_idx, device, CFG, logger)

        dl_fold_results.append(
            (fold_idx, best_f1, norm_params, ckpt_path, report, oof_probs, oof_labels))

        if oof_probs is not None and len(oof_probs) > 0:
            oof_probs_dl_list.append(oof_probs)
            oof_labels_list.append(oof_labels)
            dl_all_preds.extend(oof_probs.argmax(axis=1).tolist())
            dl_all_labels.extend(oof_labels.tolist())

    t_phaseA = time.time() - t_phaseA_start

    logger.log("\n" + "═" * 70)
    logger.log("  PHASE A SUMMARY – SE-InceptionTime Cross-Validation")
    logger.log("═" * 70)
    dl_f1_scores = [r[1] for r in dl_fold_results]
    for fi, bf1, _, _, _, _, _ in dl_fold_results:
        logger.log(f"  [DL] Fold {fi+1}:  Best Val Macro F1 = {bf1:.6f}")
    logger.log(f"\n  [DL] Mean F1 = {np.mean(dl_f1_scores):.6f}  "
               f"±  {np.std(dl_f1_scores):.6f}")
    logger.log(f"  Phase A wall-clock: {t_phaseA / 60:.1f} min")

    if len(dl_all_labels) > 0:
        cm_text = format_confusion_matrix(
            dl_all_labels, dl_all_preds, CLASS_NAMES)
        logger.log_file("\n--- [DL] Aggregate Confusion Matrix ---")
        logger.log_file(cm_text)
        logger.log("\n--- [DL] Aggregate Confusion Matrix ---")
        logger.log(cm_text)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE B – Train LightGBM (flat 6-class)
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE B – LightGBM Training (Flat 6-Class, 100+ Tabular Features)")
    logger.log("═" * 70)

    t_phaseB_start = time.time()
    gbdt_fold_results = []
    gbdt_all_preds    = []
    gbdt_all_labels   = []

    for fold_idx in folds_to_run:
        gbdt_val_f1, gbdt_model, oof_probs_g, oof_labels_g = train_fold_gbdt(
            fold_idx, CFG, logger)

        gbdt_fold_results.append(
            (fold_idx, gbdt_val_f1, gbdt_model, oof_probs_g, oof_labels_g))

        if oof_probs_g is not None and len(oof_probs_g) > 0:
            oof_probs_gbdt_list.append(oof_probs_g)
            gbdt_all_preds.extend(oof_probs_g.argmax(axis=1).tolist())
            gbdt_all_labels.extend(oof_labels_g.tolist())

    t_phaseB = time.time() - t_phaseB_start

    logger.log("\n" + "═" * 70)
    logger.log("  PHASE B SUMMARY – LightGBM Cross-Validation")
    logger.log("═" * 70)
    gbdt_f1_scores = [r[1] for r in gbdt_fold_results]
    for fi, bf1, _, _, _ in gbdt_fold_results:
        logger.log(f"  [GBDT] Fold {fi+1}:  Val Macro F1 = {bf1:.6f}")
    if len(gbdt_f1_scores) > 0:
        logger.log(f"\n  [GBDT] Mean F1 = {np.mean(gbdt_f1_scores):.6f}  "
                   f"±  {np.std(gbdt_f1_scores):.6f}")
    logger.log(f"  Phase B wall-clock: {t_phaseB / 60:.1f} min")
    logger.log(f"  Total A+B wall-clock: {(t_phaseA + t_phaseB) / 60:.1f} min")

    if len(gbdt_all_labels) > 0:
        cm_text_g = format_confusion_matrix(
            gbdt_all_labels, gbdt_all_preds, CLASS_NAMES)
        logger.log_file("\n--- [GBDT] Aggregate Confusion Matrix ---")
        logger.log_file(cm_text_g)
        logger.log("\n--- [GBDT] Aggregate Confusion Matrix ---")
        logger.log(cm_text_g)

    # ════════════════════════════════════════════════════════════════════════
    # OOF BLENDING WEIGHT OPTIMIZATION
    # ════════════════════════════════════════════════════════════════════════
    blend_weights = np.full(NUM_CLASSES, 0.5, dtype=np.float64)  # default fallback

    if (len(oof_probs_dl_list) > 0 and len(oof_probs_gbdt_list) > 0 and
            len(oof_labels_list) > 0):

        # Concatenate OOF matrices from all folds
        oof_dl_all   = np.concatenate(oof_probs_dl_list,   axis=0)  # (N_total, 6)
        oof_gbdt_all = np.concatenate(oof_probs_gbdt_list, axis=0)  # (N_total, 6)
        oof_lbl_all  = np.concatenate(oof_labels_list,     axis=0)  # (N_total,)

        # Verify alignment: DL and GBDT OOF may differ in size if folds differ
        # Use minimum size for safe alignment
        n_min = min(len(oof_dl_all), len(oof_gbdt_all), len(oof_lbl_all))
        if n_min < len(oof_lbl_all):
            logger.log(f"  NOTE: Aligning OOF sizes: DL={len(oof_dl_all)}  "
                       f"GBDT={len(oof_gbdt_all)}  Using N={n_min}")
        oof_dl_all   = oof_dl_all[:n_min]
        oof_gbdt_all = oof_gbdt_all[:n_min]
        oof_lbl_all  = oof_lbl_all[:n_min]

        blend_weights = optimize_blend_weights(
            oof_dl_all, oof_gbdt_all, oof_lbl_all, logger)
    else:
        logger.log("\n  WARNING: Insufficient OOF data for weight optimization. "
                   "Using equal blend W=[0.5, ...].")

    # Log final blend weights
    logger.log("\n" + "─" * 70)
    logger.log("  Final Blend Weight Vector W (used for test inference):")
    for c in range(NUM_CLASSES):
        logger.log(f"    w_{c} ({CLASS_NAMES[c]}): {blend_weights[c]:.6f}")
    logger.log("─" * 70)

    # ════════════════════════════════════════════════════════════════════════
    # INFERENCE – Flat Blending with TTFN
    # ════════════════════════════════════════════════════════════════════════
    if len(dl_fold_results) > 0:
        submission = flat_ensemble_inference(
            dl_fold_results   = dl_fold_results,
            gbdt_fold_results = gbdt_fold_results,
            blend_weights     = blend_weights,
            cfg               = CFG,
            device            = device,
            logger            = logger,
        )

    logger.log("\n" + "=" * 70)
    logger.log("  Pipeline v10 (SE-InceptionTime + LightGBM + OOF Blend + TTFN) complete.")
    logger.log(f"  Blend weight vector W: {np.round(blend_weights, 4).tolist()}")
    logger.log(f"  Submission: {CFG['submission_path']}")
    logger.log("=" * 70)

    logger.close()


if __name__ == "__main__":
    main()
