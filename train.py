"""
train.py
--------
HAR Flat 6-Class OOF Stacking & LightGBM Meta-Learner Pipeline (v12).

Architecture
────────────
  Global Flat 6-Class Task  : L0, L1, L2, L3, L4, L5
  Base Model  (DL)          : SE-InceptionTime1D – 6-class flat classifier
  Meta-Learner (GBDT)       : LightGBM on 142-dimensional stacked meta-features
                               = 6 CNN OOF probabilities + 136 handcrafted tabular features

Training Phases
───────────────
  Phase A – SE-InceptionTime Base Model (OOF Feature Extraction):
    • 5-fold StratifiedGroupKFold on flat 6-class labels
    • Global Z-score normalisation computed from the ENTIRE training set (not per-fold)
    • Focal Loss (γ=2.0, ε=0.1 label smoothing, α=inverse-frequency)
    • Batch-level Mixup (Beta(α=0.2))
    • AdamW + CosineAnnealingLR
    • Early stop on validation Macro F1 (patience=25)
    • Saves: checkpoints/v12_fold_{n}_best.pth
    • OOF Collection:
        – At best epoch of each fold, stores val Softmax probs → (N_train, 6) OOF matrix
        – For test set, runs all 5 fold weights × TTA(0.95, 1.00, 1.05) → (N_test, 6) mean

  Phase B – Post-Normalisation Feature Engineering (v12 Surgery):
    • Extract 136 handcrafted temporal features per signal window (FFT PRUNED)
    • ALWAYS applied AFTER global Z-score normalisation (no data leakage)
    • Features: 17 per channel × 8 channels = 136
    • v12 changes: FFT features REMOVED; Jerk statistics + Rolling Variation CV INJECTED

  Phase C – LightGBM Stacked Meta-Learner (v12 Upgrade):
    • Concatenate CNN OOF probs (N, 6) + tabular features (N, 136) → (N, 142)
    • Train 5-fold 6-class LightGBM classifier over the 142-dimensional meta-space
    • Custom Multi-Class Focal Loss (γ=2.0) replaces default multiclass objective
    • colsample_bytree=0.6 feature fraction defence prevents CNN prob saturation
    • Clean argmax over meta-learner output → submission_v12_stacking.csv

    REMOVED from v11 (v12 decommissions):
      ✗ Test-Time Feature Normalization (TTFN) – completely removed
      ✗ Manual probability blending (0.5×CNN + 0.5×GBDT)
      ✗ OOF blending weight optimization (scipy L-BFGS-B)
      ✗ Default objective='multiclass' + class_weight='balanced' (replaced by focal loss)
      ✗ FFT frequency-domain features (fft_dc, fft_max_amp, fft_energy) per channel

    ADDED in v12:
      ✓ Custom LightGBM Multi-Class Focal Loss objective (γ=2.0)
      ✓ Jerk statistics: mean & std of d²X/dt² (2nd temporal derivative)
      ✓ Rolling variance Coefficient of Variation (CV = std/mean over 50-step windows)
      ✓ Feature fraction defense: colsample_bytree=0.6

Usage
─────
    conda activate PyTorch
    python train.py               # fold 1 only
    python train.py --all-folds   # full 5-fold CV + stacking inference

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
    raise ImportError(
        "LightGBM is not installed. Please install: pip install lightgbm")

from dataset import (
    load_all_samples, load_test_samples, get_fold_splits,
    normalize, compute_normalization_params, HARDataset,
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
    "submission_path"   : "./submission_v12_stacking.csv",
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
    "nb_filters"        : 64,      # output channels per branch per block
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
    "run_all_folds"     : True,    # v12 default: always run all 5 folds

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
# Module 4: Temporal Feature Extractor v12 (136 features, FFT pruned)
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


def _rolling_variance_cv(signal_1d: np.ndarray, window: int = 50) -> float:
    """
    Rolling Variance Coefficient of Variation (CV = std / |mean|).

    Slides a window of `window` steps across the signal (stride=1), computes
    the variance at each position, then returns CV = std(vars) / |mean(vars)|.
    Distinguishes irregular L2 stair-climbing (high CV) from uniform L1/L4
    running cadences (low CV).  Returns 0.0 when mean ≈ 0.
    """
    n = len(signal_1d)
    n_windows = max(n - window + 1, 1)
    rolling_vars = np.array(
        [float(np.var(signal_1d[s:s + window])) for s in range(n_windows)],
        dtype=np.float64,
    )
    rv_mean = float(np.mean(rolling_vars))
    rv_std  = float(np.std(rolling_vars))
    if abs(rv_mean) < 1e-12:
        return 0.0
    return rv_std / abs(rv_mean)


def extract_tabular_features(signals_np: np.ndarray) -> np.ndarray:
    """
    Extract 136 handcrafted temporal features from (N, 8, 300) signal windows.

    IMPORTANT: This function MUST be called on NORMALISED signals (post global Z-score).

    v12 Feature layout per channel (17 features × 8 channels = 136 total):
    ─────────────────────────────────────────────────────────────────────────
    Time-Domain Statistics (10 features):
        mean, std, median, max, min, variance, skewness, kurtosis, Q1, Q3

    Time-Domain Dynamics – 1st derivative (4 features):
        deriv_mean, deriv_std, deriv_zcr, deriv_energy

    Jerk Statistics – 2nd derivative d²X/dt² (2 features):
        jerk_mean  – mean of |d²X/dt²|, maps sudden structural impacts
        jerk_std   – std of d²X/dt²,   quantifies jolt irregularity

    Rolling Variation Complexity (1 feature):
        rolling_var_cv – CV of 50-step rolling variance (std/|mean|);
                         high for irregular L2 stair motion, low for periodic L1/L4

    v12 PRUNED (vs v11):
        ✗ fft_dc_component, fft_max_amplitude, fft_spectral_energy  – all removed

    Total: 17 × 8 = 136 features per sample.

    Parameters
    ----------
    signals_np : np.ndarray, shape (N, 8, 300)
                 MUST be Z-score normalised (training-set reference).

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
            feat_mean   = float(np.mean(sig))
            feat_std    = float(np.std(sig))
            feat_median = float(np.median(sig))
            feat_max    = float(np.max(sig))
            feat_min    = float(np.min(sig))
            feat_var    = float(np.var(sig))
            feat_skew   = float(scipy_skew(sig, bias=True))
            feat_kurt   = float(scipy_kurtosis(sig, fisher=True, bias=True))
            feat_q25    = float(np.percentile(sig, 25))
            feat_q75    = float(np.percentile(sig, 75))

            # ── 1st derivative dynamics (4 features) ──────────────────────────
            deriv        = np.diff(sig)                    # (299,)
            deriv_mean   = float(np.mean(deriv))
            deriv_std    = float(np.std(deriv))
            deriv_zcr    = _zero_crossing_rate(deriv)
            deriv_energy = float(np.sum(deriv ** 2))

            # ── Jerk: 2nd derivative d²X/dt² (2 features) ─────────────────────
            jerk      = np.diff(deriv)                     # (298,)
            jerk_mean = float(np.mean(np.abs(jerk)))       # mean absolute jerk
            jerk_std  = float(np.std(jerk))                # std of signed jerk

            # ── Rolling Variation CV (1 feature) ──────────────────────────────
            rolling_var_cv = _rolling_variance_cv(sig, window=50)

            # Collect all 17 features for this channel
            # (10 time-domain + 4 deriv + 2 jerk + 1 rolling_var_cv = 17)
            ch_feats = [
                feat_mean, feat_std, feat_median, feat_max, feat_min,
                feat_var, feat_skew, feat_kurt, feat_q25, feat_q75,
                deriv_mean, deriv_std, deriv_zcr, deriv_energy,
                jerk_mean, jerk_std,
                rolling_var_cv,
            ]
            row_feats.extend(ch_feats)

        feature_rows.append(np.array(row_feats, dtype=np.float32))

    return np.stack(feature_rows, axis=0)   # (N, 136)


N_TAB_FEATURES = 136   # 17 features × 8 ch  (v12: 10 stat + 4 deriv + 2 jerk + 1 rv_cv)
N_META_FEATURES = N_TAB_FEATURES + NUM_CLASSES   # 136 + 6 = 142


# ══════════════════════════════════════════════════════════════════════════════
# Module 5: LightGBM Meta-Learner v12 (Custom Focal Loss + Feature Fraction)
# ══════════════════════════════════════════════════════════════════════════════

# ── Custom Multi-Class Focal Loss for LightGBM ────────────────────────────────
# LightGBM custom objective receives raw leaf scores (logits) and must return
# (gradient, hessian) arrays of shape (N * num_class,) in row-major order
# [sample_0_class_0, sample_0_class_1, ..., sample_N_class_K].
#
# Focal Loss reduction for class k at sample i:
#   p_k   = softmax(z)_k
#   FL_k  = -(1 - p_true)^γ * log(p_true)      [for the true class]
#
# Compact gradient derivation (Lin et al. 2017 adapted to multi-class softmax):
#   ∂FL/∂z_k = focal_w * (p_k - 1_{y=k})
#             + γ * focal_w * log(p_true) * (p_k - 1_{y=k})  [correction term]
# where focal_w = (1 - p_true)^γ
# ─────────────────────────────────────────────────────────────────────────────

_LGBM_FOCAL_GAMMA: float = 2.0   # γ – focusing parameter


def _focal_multiclass_objective(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> tuple:
    """
    Custom multi-class Focal Loss objective for LightGBM (sklearn API).

    LightGBM's sklearn wrapper calls the custom objective as:
        grad, hess = func(y_true, y_pred)
    where y_true is a numpy array of integer labels (NOT a lgb.Dataset).

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)       – integer class labels 0..K-1
    y_pred : np.ndarray, shape (N * K,)   – raw leaf scores, row-major order

    Returns
    -------
    grad : np.ndarray, shape (N * K,)
    hess : np.ndarray, shape (N * K,)
    """
    gamma  = _LGBM_FOCAL_GAMMA
    y_true = y_true.astype(np.int32)
    N      = len(y_true)
    K      = NUM_CLASSES

    # Reshape to (N, K) and compute numerically stable softmax
    scores = y_pred.reshape(N, K)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_s  = np.exp(scores)
    prob   = exp_s / exp_s.sum(axis=1, keepdims=True)          # (N, K)

    # One-hot encoding of ground-truth labels
    one_hot = np.zeros((N, K), dtype=np.float64)
    one_hot[np.arange(N), y_true] = 1.0

    # True-class probability and focal weight per sample
    p_true  = prob[np.arange(N), y_true]                       # (N,)
    focal_w = (1.0 - p_true) ** gamma                          # (N,)
    log_pt  = np.log(p_true + 1e-9)                            # (N,)

    # ── Gradient: focal-weighted CE gradient + γ-correction term ──────────────
    # Shape (N, K): for each class k, residual = (p_k - 1_{y=k})
    residual = prob - one_hot                                   # (N, K)
    grad = (
        focal_w[:, None] * residual
        + gamma * focal_w[:, None] * log_pt[:, None] * residual
    )

    # ── Hessian: diagonal approximation, re-weighted CE hessian ─────────────
    hess = np.maximum(focal_w[:, None] * prob * (1.0 - prob), 1e-6)

    return grad.flatten().astype(np.float64), hess.flatten().astype(np.float64)


def _focal_multiclass_eval(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> tuple:
    """
    Custom Focal Loss evaluation metric (sklearn API).
    Returns ('focal_loss', value, is_higher_better=False).
    """
    gamma  = _LGBM_FOCAL_GAMMA
    y_true = y_true.astype(np.int32)
    N      = len(y_true)
    K      = NUM_CLASSES

    scores = y_pred.reshape(N, K)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_s  = np.exp(scores)
    prob   = exp_s / exp_s.sum(axis=1, keepdims=True)

    p_true  = prob[np.arange(N), y_true]
    focal_w = (1.0 - p_true) ** gamma
    loss    = float(np.mean(-focal_w * np.log(p_true + 1e-9)))
    return "focal_loss", loss, False


def build_meta_lgbm_model() -> lgb.LGBMClassifier:
    """
    Instantiate a 6-class LightGBM meta-learner for the stacking layer (v12).

    v12 upgrades vs v11:
    ──────────────────────────────────────────────────────────────────
    • objective     : custom focal loss (γ=2.0)  ← replaces 'multiclass'
    • class_weight  : REMOVED (focal loss handles imbalance natively)
    • colsample_bytree : 0.6  ← feature fraction defense (was 0.8)
    """
    return lgb.LGBMClassifier(
        objective        = _focal_multiclass_objective,
        num_class        = NUM_CLASSES,
        n_estimators     = 1000,
        learning_rate    = 0.05,
        max_depth        = 7,
        num_leaves       = 63,
        min_child_samples= 20,
        subsample        = 0.8,
        subsample_freq   = 1,
        colsample_bytree = 0.6,   # v12: feature fraction defense (was 0.8)
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        # class_weight removed – focal loss targets imbalance directly
        random_state     = 42,
        verbose          = -1,
        n_jobs           = -1,
    )


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
# Phase A: Per-fold SE-InceptionTime training + OOF extraction
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["L0", "L1", "L2", "L3", "L4", "L5"]


def train_fold_dl(fold_idx:    int,
                  device:      torch.device,
                  cfg:         dict,
                  logger:      DualLogger,
                  global_mean: np.ndarray,
                  global_std:  np.ndarray,
                  all_signals: np.ndarray,
                  all_labels:  np.ndarray,
                  all_groups:  np.ndarray):
    """
    Train one CV fold for the flat 6-class SE-InceptionTime.

    Uses GLOBAL training-set normalisation parameters (not fold-specific).

    Returns
    -------
    best_val_f1     : float
    checkpoint_path : str
    best_val_report : str
    oof_probs       : np.ndarray (N_val, 6)  – OOF probability matrix
    oof_labels      : np.ndarray (N_val,)    – true labels for this fold
    val_idx         : np.ndarray (N_val,)    – absolute index positions in all_signals
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase A] --- Fold {fold_num}/{N_FOLDS} SE-InceptionTime Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE A | FOLD {fold_num} / {N_FOLDS}  "
                    f"– Flat 6-Class SE-InceptionTime (Global Z-score Norm)")
    logger.log_file("=" * 70)

    # ── Split indices ──────────────────────────────────────────────────────────
    train_idx, val_idx = get_fold_splits(all_signals, all_labels, all_groups, fold_idx)

    # ── Apply GLOBAL normalisation (same reference for all folds) ─────────────
    # signals are (N, 300, 8) → normalise to (N, 300, 8)
    train_signals_norm = normalize(all_signals[train_idx], global_mean, global_std)
    val_signals_norm   = normalize(all_signals[val_idx],   global_mean, global_std)

    train_labels_fold = all_labels[train_idx]
    val_labels_fold   = all_labels[val_idx]

    # Class counts for focal loss alpha (from this fold's training labels)
    class_counts = np.bincount(train_labels_fold, minlength=NUM_CLASSES)

    logger.log_file(f"  Training samples  : {len(train_idx)}")
    logger.log_file(f"  Validation samples: {len(val_idx)}")
    logger.log_file("  Class counts (train fold):")
    for c, cnt in enumerate(class_counts):
        logger.log_file(f"    {CLASS_NAMES[c]}: {cnt:>5d}  "
                        f"({cnt / max(class_counts.sum(), 1) * 100:.1f}%)")
    logger.log_console(f"  [DL] Train: {len(train_idx)}  Val: {len(val_idx)}  "
                       f"Classes: {class_counts.tolist()}")

    # ── Build Datasets & DataLoaders ──────────────────────────────────────────
    train_ds = HARDataset(train_signals_norm, train_labels_fold,
                          train_labels_fold, is_train=True)
    val_ds   = HARDataset(val_signals_norm,   val_labels_fold,
                          val_labels_fold,   is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
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
                              f"v12_fold_{fold_num}_best.pth")
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    best_val_preds  = (None, None)
    best_val_probs  = None
    best_val_report = ""

    # ── Training loop ─────────────────────────────────────────────────────────
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

    return (early_stop.best_score, ckpt_path,
            best_val_report, oof_probs, oof_labels, val_idx)


# ══════════════════════════════════════════════════════════════════════════════
# Test set CNN probability accumulation (Phase A inference)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def accumulate_test_cnn_probs(dl_fold_results: list,
                               normed_test:     np.ndarray,
                               cfg:             dict,
                               device:          torch.device,
                               logger:          DualLogger) -> np.ndarray:
    """
    Run test inference across all trained fold models with TTA (×0.95, ×1.00, ×1.05).
    Returns arithmetic mean of all fold/TTA outputs: (N_test, 6).

    Parameters
    ----------
    dl_fold_results : list of (fold_idx, best_f1, ckpt_path, report, oof_p, oof_l, val_idx)
    normed_test     : np.ndarray (N_test, 8, 300) – globally normalised (C, L) ordering
    """
    n_test     = normed_test.shape[0]
    prob_accum = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
    n_accum    = 0

    test_tensor    = torch.from_numpy(normed_test.astype(np.float32))
    dummy_labels   = torch.zeros(n_test, dtype=torch.long)
    test_ds        = TensorDataset(test_tensor, dummy_labels)
    test_loader    = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2,
                                shuffle=False, num_workers=cfg["num_workers"],
                                pin_memory=cfg["pin_memory"])

    for fold_idx, best_f1, ckpt_path, _, _, _, _ in dl_fold_results:
        fold_num = fold_idx + 1
        logger.log(f"    [CNN Test] Fold {fold_num}: loading {ckpt_path}  "
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
        for signals, _ in test_loader:
            x      = signals.to(device, non_blocking=True)
            # TTA: original / scale-up (+5%) / scale-down (−5%)
            p_orig = F.softmax(model(x),        dim=1).cpu().numpy()
            p_up   = F.softmax(model(x * 1.05), dim=1).cpu().numpy()
            p_down = F.softmax(model(x * 0.95), dim=1).cpu().numpy()
            p_tta  = (p_orig + p_up + p_down) / 3.0
            fold_probs.append(p_tta)

        fold_probs_arr  = np.concatenate(fold_probs, axis=0)   # (N_test, 6)
        prob_accum     += fold_probs_arr
        n_accum        += 1
        logger.log(f"      → TTA probs accumulated (fold {fold_num})")

    if n_accum > 0:
        avg_test_probs = prob_accum / n_accum   # arithmetic mean (N_test, 6)
    else:
        logger.log("  WARNING: No valid DL folds. CNN test probs default to uniform.")
        avg_test_probs = np.full((n_test, NUM_CLASSES),
                                  1.0 / NUM_CLASSES, dtype=np.float64)

    return avg_test_probs.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Phase C: LightGBM Meta-Learner Stacking
# ══════════════════════════════════════════════════════════════════════════════

def train_meta_lgbm(meta_X_train: np.ndarray,
                    meta_y_train: np.ndarray,
                    meta_X_test:  np.ndarray,
                    all_signals_train_norm_CL: np.ndarray,
                    all_labels:   np.ndarray,
                    folds_to_run: list,
                    all_groups:   np.ndarray,
                    cfg:          dict,
                    logger:       DualLogger) -> np.ndarray:
    """
    Train a 5-fold LightGBM meta-learner on the 142-dimensional stacked meta-space.

    Parameters
    ----------
    meta_X_train : (N_train, 142) – CNN OOF probs (6) + tabular features (136)
    meta_y_train : (N_train,)     – true labels
    meta_X_test  : (N_test, 142)  – CNN test probs (6) + tabular features (136)
    all_signals_train_norm_CL : (N_train, 8, 300) – for group structure reference
    all_labels   : (N_train,)     – aligned labels (same ordering as meta_X_train)
    folds_to_run : list of fold indices
    all_groups   : (N_train,)     – user group IDs (no test set contamination)
    cfg          : dict
    logger       : DualLogger

    Returns
    -------
    test_preds_final : np.ndarray (N_test,) – argmax predictions
    """
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE C – LightGBM Meta-Learner Stacking (142-dim)")
    logger.log("═" * 70)
    logger.log(f"  Meta-feature matrix shape: train={meta_X_train.shape}  "
               f"test={meta_X_test.shape}")
    logger.log(f"  Feature breakdown: {NUM_CLASSES} CNN OOF probs + "
               f"{N_TAB_FEATURES} tabular = {N_META_FEATURES} total")

    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(sgkf.split(meta_X_train, meta_y_train, groups=all_groups))

    meta_oof_preds  = np.zeros(len(meta_y_train), dtype=np.int64)
    meta_test_probs = np.zeros((meta_X_test.shape[0], NUM_CLASSES), dtype=np.float64)
    meta_f1_scores  = []
    meta_models     = []

    folds_iter = folds_to_run if folds_to_run else list(range(N_FOLDS))

    for fold_idx in folds_iter:
        fold_num = fold_idx + 1
        tr_idx, vl_idx = splits[fold_idx]

        X_tr = meta_X_train[tr_idx]
        y_tr = meta_y_train[tr_idx]
        X_vl = meta_X_train[vl_idx]
        y_vl = meta_y_train[vl_idx]

        logger.log(f"\n  [Meta-LGB] Fold {fold_num}/{N_FOLDS}  "
                   f"train={len(tr_idx)}  val={len(vl_idx)}")

        meta_model = build_meta_lgbm_model()

        t_start = time.time()
        meta_model.fit(X_tr, y_tr)
        elapsed = time.time() - t_start

        vl_probs  = meta_model.predict_proba(X_vl)     # (N_val, 6)
        vl_preds  = vl_probs.argmax(axis=1)
        fold_f1   = f1_score(y_vl, vl_preds, average="macro", zero_division=0)
        fold_acc  = accuracy_score(y_vl, vl_preds) * 100.0

        meta_oof_preds[vl_idx] = vl_preds
        meta_f1_scores.append(fold_f1)
        meta_models.append(meta_model)

        # Accumulate test predictions from this fold
        meta_test_probs += meta_model.predict_proba(meta_X_test)

        logger.log(f"  [Meta-LGB|Fold {fold_num}] Val Macro F1: {fold_f1:.4f}  "
                   f"Acc: {fold_acc:.2f}%  Train time: {elapsed:.1f}s")
        logger.log_file(f"  [Meta-LGB|Fold {fold_num}] Val Macro F1: {fold_f1:.6f}  "
                        f"Acc: {fold_acc:.4f}%  Train time: {elapsed:.2f}s")
        logger.log_file(f"  [Meta-LGB|Fold {fold_num}] Classification Report:")
        logger.log_file(classification_report(
            y_vl, vl_preds, target_names=CLASS_NAMES,
            digits=4, zero_division=0))
        logger.log_file("─" * 70)

        # Feature importances for this fold
        fi = meta_model.feature_importances_
        # Build feature names: cnn_prob_L0..L5 + tabular features
        feat_names = [f"cnn_prob_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)]
        feat_names += [f"tab_ch{ch}_{stat}" for ch in range(8)
                       for stat in ["mean","std","median","max","min","var",
                                    "skew","kurt","q25","q75",
                                    "deriv_mean","deriv_std","deriv_zcr","deriv_energy",
                                    "jerk_mean","jerk_std","rolling_var_cv"]]
        fi_pairs = sorted(zip(feat_names, fi), key=lambda x: x[1], reverse=True)
        top20_str = "\n".join(
            f"    {rank+1:>2}. {nm:<38s} {imp:>8d}"
            for rank, (nm, imp) in enumerate(fi_pairs[:20])
        )
        logger.log_file(f"\n  [Meta-LGB|Fold {fold_num}] Top-20 Feature Importances "
                        f"(gain split):\n{top20_str}")

    # ── Phase C summary ───────────────────────────────────────────────────────
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE C SUMMARY – LightGBM Meta-Learner CV")
    logger.log("═" * 70)
    for fi_idx, f1_val in enumerate(meta_f1_scores):
        logger.log(f"  [Meta-LGB] Fold {folds_iter[fi_idx]+1}:  Val Macro F1 = {f1_val:.6f}")
    logger.log(f"\n  [Meta-LGB] Mean F1 = {np.mean(meta_f1_scores):.6f}  "
               f"±  {np.std(meta_f1_scores):.6f}")

    # OOF F1 (full meta-OOF, only meaningful when all 5 folds are run)
    oof_f1 = f1_score(meta_y_train, meta_oof_preds, average="macro", zero_division=0)
    logger.log(f"  [Meta-LGB] OOF Macro F1 (all folds combined): {oof_f1:.6f}")

    if len(meta_all_labels := meta_y_train) > 0:
        cm_text = format_confusion_matrix(
            meta_y_train.tolist(), meta_oof_preds.tolist(), CLASS_NAMES)
        logger.log_file("\n--- [Meta-LGB] OOF Confusion Matrix ---")
        logger.log_file(cm_text)
        logger.log("\n--- [Meta-LGB] OOF Confusion Matrix ---")
        logger.log(cm_text)

    # ── Aggregate test predictions (mean over trained folds) ──────────────────
    n_folds_used = max(len(folds_iter), 1)
    avg_test_meta_probs = meta_test_probs / n_folds_used   # (N_test, 6)
    test_preds_final    = avg_test_meta_probs.argmax(axis=1).astype(np.int64)

    # ── Log aggregate feature importances across all meta-folds ──────────────
    logger.log("\n")
    logger.log("  [Meta-LGB] Aggregate Feature Importances (mean across folds):")
    feat_names_base = [f"cnn_prob_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)]
    feat_names_base += [f"tab_ch{ch}_{stat}" for ch in range(8)
                        for stat in ["mean","std","median","max","min","var",
                                     "skew","kurt","q25","q75",
                                     "deriv_mean","deriv_std","deriv_zcr","deriv_energy",
                                     "jerk_mean","jerk_std","rolling_var_cv"]]
    fi_sum = np.zeros(N_META_FEATURES, dtype=np.float64)
    for mm in meta_models:
        fi_sum += mm.feature_importances_.astype(np.float64)
    fi_mean   = fi_sum / max(len(meta_models), 1)
    fi_pairs_agg = sorted(zip(feat_names_base, fi_mean),
                          key=lambda x: x[1], reverse=True)
    logger.log(f"  {'Rank':<5} {'Feature Name':<42} {'Avg Importance':>14}")
    logger.log("  " + "─" * 65)
    for rank, (nm, imp) in enumerate(fi_pairs_agg[:20]):
        logger.log(f"  {rank+1:>4}. {nm:<42} {imp:>14.2f}")

    return test_preds_final


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HAR SE-InceptionTime v12 – OOF Stacking + LightGBM Focal Loss Meta-Learner")
    parser.add_argument("--fold-1-only", action="store_true",
                        help="Run fold 1 only (default: all 5 folds)")
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--batch-size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--patience",   type=int,   default=CFG["patience"])
    args = parser.parse_args()

    CFG["run_all_folds"] = not args.fold_1_only   # default True; --fold-1-only overrides
    CFG["epochs"]        = args.epochs
    CFG["batch_size"]    = args.batch_size
    CFG["lr"]            = args.lr
    CFG["patience"]      = args.patience
    CFG["T_max"]         = args.epochs

    seed_everything(CFG["seed"])

    logger = DualLogger(CFG["log_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    hidden_dim = CFG["nb_filters"] * 4
    header_lines = [
        "=" * 70,
        "  HAR SE-InceptionTime v12 – OOF Stacking + LightGBM Focal Loss Meta-Learner",
        "  (TTFN REMOVED | Global Z-score Norm | 142-dim | FFT pruned | LGB Focal γ=2.0)",
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
        f"  Normalisation : GLOBAL training-set Z-score (NO per-test TTFN)",
        f"  Meta Features : {NUM_CLASSES} CNN OOF probs + {N_TAB_FEATURES} tabular "
        f"= {N_META_FEATURES} dims",
        f"  Meta-Learner  : LightGBM 5-fold CV (n_estimators=1000, custom focal γ={_LGBM_FOCAL_GAMMA}, colsample=0.6)",
        f"  Tabular feat  : 17 per ch × 8 (10stat+4deriv+2jerk+1rv_cv, FFT pruned)",
        f"  CNN TTA       : ×0.95 / ×1.00 / ×1.05 arithmetic mean",
        f"  Blend logic   : NONE (meta-learner argmax only, no manual weighting)",
        f"  Epochs        : {CFG['epochs']}",
        f"  Batch         : {CFG['batch_size']}",
        f"  LR            : {CFG['lr']}",
        f"  Weight Decay  : {CFG['weight_decay']}",
        f"  Focal γ       : {CFG['focal_gamma']}",
        f"  Label Smooth  : ε={CFG['label_smoothing']}",
        f"  Mixup α       : {CFG['mixup_alpha']}",
        f"  Patience      : {CFG['patience']}",
        f"  Folds         : {'all 5' if CFG['run_all_folds'] else 'fold 1 only'}",
        f"  Submission    : {CFG['submission_path']}",
        "=" * 70,
    ]
    for line in header_lines:
        logger.log(line)

    folds_to_run = list(range(N_FOLDS)) if CFG["run_all_folds"] else [0]

    # ════════════════════════════════════════════════════════════════════════
    # STEP 0: Load ALL training samples & compute GLOBAL normalisation
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  STEP 0 – Global Data Loading & Training-Set Reference Normalisation")
    logger.log("═" * 70)

    t0_start = time.time()
    all_signals, all_labels, all_groups, all_file_ids = load_all_samples(CFG["train_root"])

    # Compute global Z-score statistics from the complete training set
    # This is the SINGLE reference scaler applied to ALL splits (train/val/test)
    global_mean, global_std = compute_normalization_params(all_signals)

    logger.log(f"  Total training samples loaded: {len(all_signals)}")
    logger.log(f"  Signal shape: {all_signals.shape}  (N, timesteps, channels)")
    logger.log(f"  Global mean  (8 channels): {np.round(global_mean, 6).tolist()}")
    logger.log(f"  Global std   (8 channels): {np.round(global_std,  6).tolist()}")
    logger.log(f"  Label distribution:")
    for c in range(NUM_CLASSES):
        cnt = int((all_labels == c).sum())
        logger.log(f"    {CLASS_NAMES[c]}: {cnt:>5d}  "
                   f"({cnt / len(all_labels) * 100:.1f}%)")

    # Load test samples (NOT normalised yet)
    raw_test_signals, test_file_ids = load_test_samples(CFG["test_root"])
    n_test = raw_test_signals.shape[0]
    logger.log(f"\n  Test samples loaded: {n_test}  shape: {raw_test_signals.shape}")

    # Apply GLOBAL training-set normalisation to test signals
    # raw_test_signals is (N, 300, 8), normalize returns (N, 300, 8)
    normed_test_signals = normalize(raw_test_signals, global_mean, global_std)

    # Transpose to (N, 8, 300) for CNN input format
    normed_test_CL = normed_test_signals.transpose(0, 2, 1).astype(np.float32)

    logger.log(f"  Test signals normalised using global training-set parameters.")
    logger.log(f"  Data loading & normalisation elapsed: "
               f"{time.time() - t0_start:.1f}s")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A – Train SE-InceptionTime (flat 6-class) & Extract OOF Probs
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE A – SE-InceptionTime Training (Flat 6-Class) + OOF Extraction")
    logger.log("═" * 70)

    t_phaseA_start = time.time()
    dl_fold_results = []

    # OOF probability matrix for the entire training set: (N_train, 6)
    # Indexed by absolute position in all_signals
    oof_cnn_probs_full  = np.zeros((len(all_signals), NUM_CLASSES), dtype=np.float32)
    oof_idx_covered     = np.zeros(len(all_signals), dtype=bool)

    dl_all_preds  = []
    dl_all_labels = []

    for fold_idx in folds_to_run:
        (best_f1, ckpt_path, report,
         oof_probs, oof_labels, val_idx) = train_fold_dl(
            fold_idx, device, CFG, logger,
            global_mean, global_std,
            all_signals, all_labels, all_groups,
        )

        dl_fold_results.append(
            (fold_idx, best_f1, ckpt_path, report, oof_probs, oof_labels, val_idx))

        # Fill OOF probability slots at the validation indices
        if oof_probs is not None and len(oof_probs) > 0:
            oof_cnn_probs_full[val_idx] = oof_probs
            oof_idx_covered[val_idx]    = True
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
    logger.log(f"  OOF slots filled: {oof_idx_covered.sum()} / {len(all_signals)}")
    logger.log(f"  Phase A wall-clock: {t_phaseA / 60:.1f} min")

    if len(dl_all_labels) > 0:
        cm_text = format_confusion_matrix(
            dl_all_labels, dl_all_preds, CLASS_NAMES)
        logger.log_file("\n--- [DL] Aggregate OOF Confusion Matrix ---")
        logger.log_file(cm_text)
        logger.log("\n--- [DL] Aggregate OOF Confusion Matrix ---")
        logger.log(cm_text)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A (cont.) – Accumulate Test CNN Probabilities
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "─" * 70)
    logger.log("  PHASE A (Test) – Accumulating CNN Test Probabilities (TTA × 5 folds)")
    logger.log("─" * 70)

    t_cnn_test = time.time()
    cnn_test_probs = accumulate_test_cnn_probs(
        dl_fold_results, normed_test_CL, CFG, device, logger)
    logger.log(f"  CNN test probs aggregated: {cnn_test_probs.shape}  "
               f"({time.time() - t_cnn_test:.1f}s)")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE B – Post-Normalisation Tabular Feature Extraction
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log(f"  PHASE B – Post-Normalisation Tabular Feature Extraction "
               f"({N_TAB_FEATURES} features per sample)")
    logger.log("═" * 70)

    t_phaseB_start = time.time()

    # all_signals is (N_train, 300, 8) → normalise → transpose to (N, 8, 300) for extractor
    logger.log("  Normalising full training set for feature extraction ...")
    all_signals_norm    = normalize(all_signals, global_mean, global_std)  # (N, 300, 8)
    all_signals_norm_CL = all_signals_norm.transpose(0, 2, 1).astype(np.float32)  # (N, 8, 300)

    logger.log(f"  Extracting tabular features from training set "
               f"(shape: {all_signals_norm_CL.shape}) ...")
    t_feat_tr = time.time()
    tab_feats_train = extract_tabular_features(all_signals_norm_CL)   # (N_train, 136)
    logger.log(f"  Train tabular features extracted: {tab_feats_train.shape}  "
               f"({time.time() - t_feat_tr:.1f}s)")

    # Test: normed_test_signals is (N_test, 300, 8) → transpose to (N, 8, 300)
    normed_test_CL_for_feat = normed_test_signals.transpose(0, 2, 1).astype(np.float32)
    logger.log(f"  Extracting tabular features from test set "
               f"(shape: {normed_test_CL_for_feat.shape}) ...")
    t_feat_te = time.time()
    tab_feats_test  = extract_tabular_features(normed_test_CL_for_feat)   # (N_test, 136)
    logger.log(f"  Test  tabular features extracted: {tab_feats_test.shape}  "
               f"({time.time() - t_feat_te:.1f}s)")

    t_phaseB = time.time() - t_phaseB_start
    logger.log(f"  Phase B wall-clock: {t_phaseB:.1f}s")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE C – Build 142-dim Meta-Feature Matrices & Train LightGBM
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log(f"  PHASE C – Stacking Meta-Feature Construction "
               f"(CNN probs {NUM_CLASSES} + tabular {N_TAB_FEATURES} = {N_META_FEATURES})")
    logger.log("═" * 70)

    # For training: use only the OOF-covered samples (all samples when all folds run)
    train_mask = oof_idx_covered   # boolean (N_train,)
    n_meta_train = train_mask.sum()

    if n_meta_train == 0:
        logger.log("  ERROR: No OOF-covered samples found. "
                   "Run with --all-folds for a proper stacking pipeline.")
        logger.close()
        return

    # Concatenate: (N_covered, 6) CNN probs + (N_covered, 136) tabular = (N_covered, 142)
    meta_X_train = np.concatenate([
        oof_cnn_probs_full[train_mask],   # (N, 6)
        tab_feats_train[train_mask],       # (N, 136)
    ], axis=1).astype(np.float32)          # (N, 142)

    meta_y_train = all_labels[train_mask]  # (N,)
    meta_groups_train = all_groups[train_mask]  # (N,) – for meta-fold splits

    # Test: (N_test, 6) CNN avg probs + (N_test, 136) tabular = (N_test, 142)
    meta_X_test = np.concatenate([
        cnn_test_probs,   # (N_test, 6)
        tab_feats_test,   # (N_test, 136)
    ], axis=1).astype(np.float32)

    logger.log(f"  Meta-train matrix: {meta_X_train.shape}  ({n_meta_train} samples)")
    logger.log(f"  Meta-test  matrix: {meta_X_test.shape}")
    logger.log(f"  Meta-label distribution:")
    for c in range(NUM_CLASSES):
        cnt = int((meta_y_train == c).sum())
        logger.log(f"    {CLASS_NAMES[c]}: {cnt:>5d}  "
                   f"({cnt / max(len(meta_y_train), 1) * 100:.1f}%)")

    t_phaseC_start = time.time()
    test_preds_final = train_meta_lgbm(
        meta_X_train         = meta_X_train,
        meta_y_train         = meta_y_train,
        meta_X_test          = meta_X_test,
        all_signals_train_norm_CL = all_signals_norm_CL[train_mask],
        all_labels           = meta_y_train,
        folds_to_run         = folds_to_run,
        all_groups           = meta_groups_train,
        cfg                  = CFG,
        logger               = logger,
    )
    t_phaseC = time.time() - t_phaseC_start
    logger.log(f"\n  Phase C wall-clock: {t_phaseC / 60:.1f} min")

    # ════════════════════════════════════════════════════════════════════════
    # SUBMISSION ASSEMBLY
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  SUBMISSION ASSEMBLY")
    logger.log("═" * 70)

    submission = (
        pd.DataFrame({"Id": test_file_ids, "Label": test_preds_final})
        .sort_values("Id")
        .reset_index(drop=True)
    )
    submission.to_csv(CFG["submission_path"], index=False)

    logger.log(f"  Total test samples    : {n_test}")
    logger.log(f"  Submission saved to   : {CFG['submission_path']}")
    pred_dist = submission["Label"].value_counts().sort_index()
    logger.log("  Final predicted label distribution:")
    for lbl, cnt in pred_dist.items():
        lbl_name = CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else str(lbl)
        logger.log(f"    {lbl_name} (Label {lbl}): {cnt:>5d}  "
                   f"({cnt / n_test * 100:.1f}%)")

    total_time = t_phaseA + t_phaseB + t_phaseC
    logger.log("\n" + "=" * 70)
    logger.log("  Pipeline v12 (SE-InceptionTime OOF Stacking + LightGBM Focal Loss) complete.")
    logger.log(f"  Phase A (CNN Training):          {t_phaseA / 60:.1f} min")
    logger.log(f"  Phase B (Feature Extraction):    {t_phaseB:.1f}s")
    logger.log(f"  Phase C (Meta-Learner Training): {t_phaseC / 60:.1f} min")
    logger.log(f"  Total wall-clock:                {total_time / 60:.1f} min")
    logger.log(f"  Submission: {CFG['submission_path']}")
    logger.log("=" * 70)

    logger.close()


if __name__ == "__main__":
    main()
