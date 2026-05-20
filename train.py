"""
train.py
--------
End-to-end training pipeline for the HAR 1D-ResNet (v6).

Features
────────
  • Deterministic seeding via seed_everything(42) for absolute reproducibility
  • 5-fold StratifiedGroupKFold cross-validation
  • Uniform Focal Loss (γ=2.0 for ALL classes) + Label Smoothing (ε=0.1)
    with α=inverse-frequency weights
  • Batch-level Mixup data augmentation (Beta(α=0.2, α=0.2)) in training loop
  • AdamW optimiser (weight_decay=1e-3) + CosineAnnealingLR scheduler
  • Early stopping monitored on Validation Macro F1-Score (patience=25)
  • Dual-stream logging: compressed single-line console + full-detail file log
  • Best checkpoint saved per fold as  fold_N_best.pth  (1-indexed)
  • Full classification report + confusion matrix at final CV summary
  • Automated 5-fold F1-Weighted TTA Ensemble inference → submission_v6_ensemble.csv

  v6 Changes
  ──────────
  • in_channels: 8 → 4  (radical feature pruning: mean_x/y/z + std_mag only)
  • base_filters: 64 → 32  (halved network width; filter cadence 32/32/64/128/256)
  • Instance mean centering injected in dataset.py before global Z-score scaling
  • Output CSV:  submission_v5_ensemble.csv → submission_v6_ensemble.csv

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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

from dataset import (
    build_fold_datasets, build_test_dataset,
    load_all_samples, get_fold_splits,
    N_FOLDS, N_CLASSES
)
from model import ResNet1D

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# Module 1: Deterministic Experimental Control
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42) -> None:
    """
    Lock all random number generators for absolute experimental reproducibility.

    Covers:
      • Python built-in random module
      • NumPy global RNG
      • PyTorch CPU RNG
      • PyTorch CUDA RNG (all devices)
      • cuDNN deterministic execution mode (benchmark disabled)
    """
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
    "submission_path"   : "./submission_v6_ensemble.csv",
    "checkpoint_dir"    : "./checkpoints",
    "log_path"          : "./training_log.txt",

    # Training ────────────────────────────────────────────────────────────────
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

    # Model ───────────────────────────────────────────────────────────────────
    "in_channels"       : 4,           # v6: 4 pruned channels (mean_x/y/z + std_mag)
    "num_classes"       : 6,
    "base_filters"      : 32,          # v6: halved width; cadence 32/32/64/128/256

    # Focal Loss (v5: uniform gamma across all classes)
    "focal_gamma"       : 2.0,         # uniform γ=2.0 for ALL classes

    # Label Smoothing (v5)
    "label_smoothing"   : 0.1,         # ε=0.1

    # Mixup (v5)
    "mixup_alpha"       : 0.2,         # Beta distribution parameter α

    # Cross-validation ────────────────────────────────────────────────────────
    "run_all_folds"     : False,

    # Reproducibility ─────────────────────────────────────────────────────────
    "seed"              : 42,
}


# ══════════════════════════════════════════════════════════════════════════════
# Dual-stream logger
# ══════════════════════════════════════════════════════════════════════════════

class DualLogger:
    """
    Routes output to both the terminal (console) and a persistent text file.

    Methods
    -------
    log(msg)         – write to BOTH console and file  (default for most output)
    log_file(msg)    – write to FILE only  (verbose per-epoch details, reports)
    log_console(msg) – write to CONSOLE only  (transient status, noisy for file)
    close()          – flush and close the file handle
    """

    def __init__(self, log_path: str):
        log_dir = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(log_dir, exist_ok=True)
        # Open in write mode → overwrite at run start; all subsequent writes append
        self._fh      = open(log_path, "w", encoding="utf-8", buffering=1)
        self._console = sys.__stdout__

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, msg: str = ""):
        """Write line to both console and log file."""
        self._console.write(msg + "\n")
        self._console.flush()
        self._fh.write(msg + "\n")

    def log_file(self, msg: str = ""):
        """Write line to log file only (verbose details suppressed from console)."""
        self._fh.write(msg + "\n")

    def log_console(self, msg: str = ""):
        """Write line to console only."""
        self._console.write(msg + "\n")
        self._console.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Module 2: Focal Loss with Uniform γ and Label Smoothing  (v5)
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Alpha-weighted Focal Loss with uniform focusing parameter γ and
    integrated Label Smoothing (ε).

    Standard Focal Loss (hard targets):
        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    v5 Label-Smoothed Focal Loss:
      Ground-truth one-hot targets are smoothed before computing cross-entropy:
        y_smooth[c] = (1 - ε) + ε / C   if c == true class
        y_smooth[c] = ε / C              otherwise
      where C = num_classes and ε = label_smoothing coefficient.

      With ε=0.1 and C=6:
        positive target  = 0.9 + 0.1/6  ≈ 0.9167
        negative targets = 0.1/6        ≈ 0.01667

      The Focal modulation uses the probability of the true class (p_t) to
      weight the smoothed cross-entropy loss, preserving the adaptive
      re-weighting property of Focal Loss over the soft target distribution.

    v5 Uniform γ:
      A single scalar γ=2.0 is applied identically across all 6 classes.
      The class-specific γ=3.0 penalty for Label 2 introduced in v4 is
      abolished; the model is encouraged to embrace distributional ambiguity
      rather than forcing rigid boundaries on overlapping classes.

    Parameters
    ----------
    alpha           : torch.Tensor, shape (num_classes,)
                      Per-class balancing weights (inverse-frequency, normalised).
    gamma           : float  uniform focusing parameter for all classes (2.0)
    label_smoothing : float  label smoothing coefficient ε (0.1)
    reduction       : str  'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: torch.Tensor,
                 gamma: float           = 2.0,
                 label_smoothing: float = 0.1,
                 reduction: str         = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)   # (C,)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)  raw model output (unnormalised)
        targets : (B, C)  soft / smoothed target distribution  OR
                  (B,)    integer class labels (hard targets; smoothing applied here)

        Both integer hard-target and pre-smoothed soft-target inputs are
        supported to enable straightforward Mixup integration.
        """
        C = logits.size(1)

        # ── Build smoothed target distribution ────────────────────────────────
        if targets.dim() == 1:
            # Hard integer labels → one-hot → label-smooth
            # y_smooth[true_class] = 1 - ε + ε/C
            # y_smooth[other]      = ε / C
            with torch.no_grad():
                smooth_val    = self.label_smoothing / C
                y_smooth      = torch.full_like(logits, smooth_val)          # (B, C)
                y_smooth.scatter_(1, targets.unsqueeze(1),
                                  1.0 - self.label_smoothing + smooth_val)   # (B, C)
        else:
            # Pre-smoothed soft targets supplied (e.g., from Mixup blending)
            y_smooth = targets   # (B, C)

        # ── Focal modulation using log-softmax ────────────────────────────────
        log_probs = F.log_softmax(logits, dim=1)   # (B, C)
        probs     = log_probs.exp()                 # (B, C)

        # True-class probability p_t:  use argmax of smooth targets as proxy
        # (for pure smoothed labels its argmax == original true class)
        true_class = y_smooth.argmax(dim=1)                                  # (B,)
        pt         = probs.gather(1, true_class.unsqueeze(1)).squeeze(1)     # (B,)

        # Focal modulating factor (scalar γ, uniform across all classes)
        focal_weight = (1.0 - pt) ** self.gamma                              # (B,)

        # Per-class α weight for each sample (indexed by true class)
        alpha_t = self.alpha.gather(0, true_class)                           # (B,)

        # Cross-entropy over the smoothed distribution:  -Σ y_smooth * log_probs
        ce_smooth = -(y_smooth * log_probs).sum(dim=1)                       # (B,)

        # Final per-sample focal loss
        loss = alpha_t * focal_weight * ce_smooth                            # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_focal_loss(class_counts: np.ndarray,
                     device: torch.device,
                     gamma: float           = 2.0,
                     label_smoothing: float = 0.1) -> FocalLoss:
    """
    Construct a FocalLoss with inverse-frequency alpha weights,
    uniform γ=2.0, and label smoothing ε=0.1.

    weight_c = total / (N_CLASSES × count_c),  then normalised so mean = 1.
    """
    counts  = class_counts.astype(np.float32)
    total   = counts.sum()
    weights = total / (N_CLASSES * counts)
    weights = weights / weights.mean()
    alpha   = torch.tensor(weights, dtype=torch.float32, device=device)
    return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)


# ══════════════════════════════════════════════════════════════════════════════
# Module 3: Mixup augmentation helper  (v5)
# ══════════════════════════════════════════════════════════════════════════════

def mixup_batch(signals: torch.Tensor,
                labels: torch.Tensor,
                alpha: float,
                num_classes: int,
                label_smoothing: float,
                device: torch.device):
    """
    Perform batch-level Mixup interpolation on input sequences and their
    label-smoothed target distributions.

    Algorithm
    ---------
    1. Sample λ ~ Beta(α, α), then take max(λ, 1-λ) so λ ≥ 0.5 (optional
       symmetry enforcement — omitted here; raw Beta sample used directly).
    2. Generate a random permutation index to form the mixing pair (X_j, Y_j).
    3. Linearly blend:
         X_mixed = λ · X_i  +  (1 - λ) · X_j          (input tensor)
         Y_mixed = λ · Y_i  +  (1 - λ) · Y_j          (soft label tensor)

    The blended soft labels already incorporate Label Smoothing ε: one-hot
    targets are first smoothed (y_smooth = (1-ε)·onehot + ε/C) and then
    the two smoothed distributions are linearly combined with λ.

    Parameters
    ----------
    signals       : torch.Tensor, shape (B, C, L) – batch of input sequences
    labels        : torch.Tensor, shape (B,)       – integer hard labels
    alpha         : float                           – Beta distribution parameter
    num_classes   : int                             – number of output classes (C)
    label_smoothing : float                         – label smoothing coefficient ε
    device        : torch.device

    Returns
    -------
    x_mixed  : torch.Tensor, shape (B, C, L)   – mixed input sequences
    y_mixed  : torch.Tensor, shape (B, num_classes) – mixed smoothed targets
    lam      : float                            – the sampled mixing coefficient
    """
    B = signals.size(0)

    # ── Sample λ from Beta(α, α) ─────────────────────────────────────────────
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    # ── Random permutation for the mixing partner ─────────────────────────────
    perm = torch.randperm(B, device=device)

    # ── Build label-smoothed one-hot targets for both Xi and Xj ──────────────
    smooth_val = label_smoothing / num_classes
    y_i        = torch.full((B, num_classes), smooth_val,
                            dtype=torch.float32, device=device)
    y_i.scatter_(1, labels.unsqueeze(1),
                 1.0 - label_smoothing + smooth_val)          # (B, C)

    y_j = y_i[perm]                                           # (B, C)

    # ── Linear interpolation ─────────────────────────────────────────────────
    x_mixed = lam * signals + (1.0 - lam) * signals[perm]    # (B, C, L)
    y_mixed = lam * y_i     + (1.0 - lam) * y_j              # (B, C)

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
# Single epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device,
                    mixup_alpha: float, label_smoothing: float) -> float:
    """
    Train for one epoch with batch-level Mixup augmentation.

    For each mini-batch:
      1. Move data to device.
      2. Apply Mixup: blend (X_i, Y_i) with (X_j, Y_j) using λ ~ Beta(α, α).
      3. Forward pass on the mixed input X_mixed.
      4. Compute label-smoothed Focal Loss on mixed soft targets Y_mixed.
         Because FocalLoss.forward() accepts pre-smoothed soft targets (2D),
         the Mixup-blended targets are passed directly — no secondary smoothing.
      5. Backpropagate and clip gradients (max_norm=1.0).
    """
    model.train()
    running_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)   # (B, 8, 300)
        labels  = labels.to(device,  non_blocking=True)   # (B,)

        # ── Mixup ─────────────────────────────────────────────────────────────
        x_mixed, y_mixed, lam = mixup_batch(
            signals, labels,
            alpha=mixup_alpha,
            num_classes=N_CLASSES,
            label_smoothing=label_smoothing,
            device=device,
        )

        optimizer.zero_grad()

        # ── Forward + Loss ────────────────────────────────────────────────────
        # Pass pre-blended soft targets (2D) directly to FocalLoss.
        # FocalLoss.forward() detects dim==2 and skips the internal smoothing
        # step, evaluating the focal-weighted cross-entropy over y_mixed.
        logits = model(x_mixed)
        loss   = criterion(logits, y_mixed)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * signals.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Validation loop (no Mixup, no augmentation).
    Uses hard integer labels and FocalLoss with internal label smoothing.
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        logits  = model(signals)
        # Pass integer labels (1D) — FocalLoss applies label smoothing internally
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
# Per-fold training routine
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(fold_idx: int, device: torch.device, cfg: dict,
               logger: DualLogger):
    """
    Train one CV fold end-to-end.

    Returns
    -------
    best_val_f1     : float
    norm_params     : (mean, std)
    checkpoint_path : str
    best_val_preds  : (preds, labels) from best epoch
    best_val_report : str  – classification report text
    """

    fold_num = fold_idx + 1   # 1-indexed for display

    # ── Console: clean single-line separator ──────────────────────────────────
    logger.log(f"\n--- Fold {fold_num}/{N_FOLDS} Starting ---")

    # ── File: verbose fold header ─────────────────────────────────────────────
    logger.log_file("=" * 70)
    logger.log_file(f"FOLD {fold_num} / {N_FOLDS}  ──  Starting")
    logger.log_file("=" * 70)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds, norm_params, class_counts = build_fold_datasets(
        cfg["train_root"], fold_idx=fold_idx
    )

    # File-only dataset info
    logger.log_file(f"  Training samples  : {len(train_ds)}")
    logger.log_file(f"  Validation samples: {len(val_ds)}")
    logger.log_file("  Class counts (train fold):")
    for c, cnt in enumerate(class_counts):
        logger.log_file(f"    Label {c}: {cnt:>5d}  ({cnt / class_counts.sum() * 100:.1f}%)")

    # Console: one-liner summary
    logger.log_console(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  "
                       f"Classes: {class_counts.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResNet1D(in_channels=cfg["in_channels"],
                     num_classes=cfg["num_classes"],
                     base_filters=cfg["base_filters"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_file(f"  Model: ResNet1D  |  Trainable params: {n_params:,}")

    # ── Loss (v5: uniform γ=2.0 + label smoothing ε=0.1) ─────────────────────
    criterion = build_focal_loss(
        class_counts, device,
        gamma=cfg["focal_gamma"],
        label_smoothing=cfg["label_smoothing"],
    )
    counts    = class_counts.astype(np.float32)
    weights   = counts.sum() / (N_CLASSES * counts)
    weights   = weights / weights.mean()
    logger.log_file(
        f"  Loss: Focal Loss (γ={cfg['focal_gamma']} uniform, "
        f"ε={cfg['label_smoothing']} label smoothing, α=inverse-frequency)"
    )
    logger.log_file("  Effective alpha weights: " +
                    "  ".join(f"L{c}:{w:.3f}" for c, w in enumerate(weights)))
    logger.log_file(
        f"  Mixup: Beta(α={cfg['mixup_alpha']}, α={cfg['mixup_alpha']})  "
        f"[training only]"
    )

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])

    # ── Early stopping ────────────────────────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    # Checkpoint file uses 1-indexed fold number: fold_1_best.pth … fold_5_best.pth
    ckpt_path  = os.path.join(cfg["checkpoint_dir"], f"fold_{fold_num}_best.pth")
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_preds  = (None, None)
    best_val_report = ""

    for epoch in range(1, cfg["epochs"] + 1):
        t_start    = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=cfg["mixup_alpha"],
            label_smoothing=cfg["label_smoothing"],
        )
        scheduler.step()
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - t_start
        lr_now  = scheduler.get_last_lr()[0]

        # ── Console: single compact line matching required template ───────────
        # Template: [Fold F/5] Ep X/80 | LR: X.Xe-X | Loss: T:X.XX / V:X.XX |
        #           Val Acc: XX.XX% | Val F1: X.XXXX | Time: X.Xs
        logger.log_console(
            f"[Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.1e} | "
            f"Loss: T:{train_loss:.2f} / V:{val_loss:.2f} | "
            f"Val Acc: {val_acc:5.2f}% | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── File: same header with higher precision ───────────────────────────
        logger.log_file(
            f"[Fold {fold_num}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.4e} | "
            f"Loss: T:{train_loss:.6f} / V:{val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f}% | "
            f"Val F1: {val_f1:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Update best report
        if val_f1 >= early_stop.best_score:
            best_val_preds  = (val_preds, val_labels)
            best_val_report = classification_report(
                val_labels, val_preds, digits=4, zero_division=0)

        should_stop = early_stop.step(val_f1, model, epoch)
        if should_stop:
            msg = (f"  >> Early stopping at epoch {epoch}. "
                   f"Best epoch: {early_stop.best_epoch}  "
                   f"Best F1: {early_stop.best_score:.4f}")
            logger.log(msg)
            break

    # ── Post-fold summary ─────────────────────────────────────────────────────
    fold_summary = (
        f"--- Fold {fold_num} done | "
        f"Best Val Macro F1: {early_stop.best_score:.6f} "
        f"(epoch {early_stop.best_epoch}) ---"
    )
    logger.log(fold_summary)

    # File: full classification report after each fold
    logger.log_file("")
    logger.log_file(f"  Classification Report @ Best Epoch ({early_stop.best_epoch}):")
    logger.log_file(best_val_report)
    logger.log_file("─" * 70)

    return (early_stop.best_score, norm_params, ckpt_path,
            best_val_preds, best_val_report)


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 + 5: F1-Weighted TTA Ensemble inference  (v5, preserved from v4)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ensemble_inference(fold_results: list, cfg: dict,
                       device: torch.device,
                       logger: DualLogger) -> pd.DataFrame:
    """
    Automated 5-fold F1-Weighted TTA Ensemble inference pipeline.

    Module 4 – Test-Time Augmentation (TTA):  [PRESERVED UNCHANGED from v4]
      For each test batch, three distinct scaling profiles are evaluated through
      the active fold model, and their Softmax distributions are averaged:
        1. Original:    X
        2. Scaled Up:   X × 1.05
        3. Scaled Down: X × 0.95
      The mean of these three distributions defines that fold's output probability.

    Module 5 – F1-Weighted Softmax Averaging:  [PRESERVED UNCHANGED from v4]
      Folds are weighted by their best validation macro F1-score:
        P_ensemble = Σ(F1_i × P_i) / Σ(F1_i)
      This replaces the uniform average (Σ P_i / N) from v3.

    Parameters
    ----------
    fold_results : list of (fold_idx, best_f1, norm_params, ckpt_path, report)
                   The norm_params from the first fold entry are used to
                   normalise the test set (consistent with the submission pipeline).
    cfg          : dict – global config
    device       : torch.device
    logger       : DualLogger

    Returns
    -------
    submission : pd.DataFrame with columns ['Id', 'Label'], sorted by Id
    """
    logger.log("\n" + "=" * 70)
    logger.log("  AUTO-ENSEMBLE INFERENCE  (5-Fold F1-Weighted TTA Ensemble)")
    logger.log("=" * 70)

    # Use the norm params from the first available fold to normalise test data.
    # (All folds yield very similar norm params; fold 1 is the canonical choice.)
    first_norm = fold_results[0][2]   # (mean, std)
    test_ds, file_ids = build_test_dataset(cfg["test_root"], first_norm[0], first_norm[1])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    n_test    = len(test_ds)
    n_classes = cfg["num_classes"]

    # ── F1-weighted accumulator ───────────────────────────────────────────────
    # Accumulate F1_i × P_i for each fold; sum of F1_i is tracked separately.
    weighted_prob_sum = np.zeros((n_test, n_classes), dtype=np.float64)
    f1_weight_total   = 0.0

    for fold_idx, best_f1, norm_params, ckpt_path, _ in fold_results:
        fold_num = fold_idx + 1
        logger.log(f"  Loading fold {fold_num} checkpoint: {ckpt_path}  "
                   f"(Val F1={best_f1:.6f})")

        model = ResNet1D(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            base_filters=cfg["base_filters"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # ── Module 4: TTA per batch ───────────────────────────────────────────
        fold_probs = []
        for signals, _ in test_loader:
            x_orig = signals.to(device, non_blocking=True)          # (B, 8, 300)

            # TTA variant 1: Original
            p_orig  = F.softmax(model(x_orig),              dim=1)  # (B, C)
            # TTA variant 2: Scaled Up  (+5%)
            p_up    = F.softmax(model(x_orig * 1.05),       dim=1)  # (B, C)
            # TTA variant 3: Scaled Down (−5%)
            p_down  = F.softmax(model(x_orig * 0.95),       dim=1)  # (B, C)

            # Average the three TTA softmax distributions for this fold
            p_tta = (p_orig + p_up + p_down) / 3.0                  # (B, C)
            fold_probs.append(p_tta.cpu().numpy())

        fold_probs_arr = np.concatenate(fold_probs, axis=0)          # (N, C)

        # ── Module 5: F1-weighted accumulation ───────────────────────────────
        weighted_prob_sum += best_f1 * fold_probs_arr
        f1_weight_total   += best_f1

        logger.log(f"    Fold {fold_num} TTA probabilities accumulated  "
                   f"(shape: {fold_probs_arr.shape}, weight: {best_f1:.6f})")

    # ── Final F1-weighted ensemble probability ────────────────────────────────
    # P_ensemble = Σ(F1_i × P_i) / Σ(F1_i)
    ensemble_probs = weighted_prob_sum / f1_weight_total              # (N, C)
    ensemble_preds = ensemble_probs.argmax(axis=1)                    # (N,)

    # Build submission DataFrame
    submission = (
        pd.DataFrame({"Id": file_ids, "Label": ensemble_preds})
        .sort_values("Id")
        .reset_index(drop=True)
    )

    # Save
    submission.to_csv(cfg["submission_path"], index=False)

    logger.log(f"\n  Ensemble folds used        : {len(fold_results)}")
    logger.log(f"  Total F1 weight            : {f1_weight_total:.6f}")
    logger.log(f"  Total test samples         : {n_test}")
    logger.log(f"  Submission saved to        : {cfg['submission_path']}")
    pred_dist = submission["Label"].value_counts().sort_index()
    logger.log("  Predicted label distribution on test set (ensemble):")
    for lbl, cnt in pred_dist.items():
        logger.log(f"    Label {lbl}: {cnt:>5d}  ({cnt / n_test * 100:.1f}%)")

    return submission


# ══════════════════════════════════════════════════════════════════════════════
# Test-set inference (single checkpoint) — retained for diagnostic use
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_test(checkpoint_path: str, norm_params,
                 cfg: dict, device: torch.device) -> pd.DataFrame:
    mean, std = norm_params
    test_ds, _ = build_test_dataset(cfg["test_root"], mean, std)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2,
                             shuffle=False, num_workers=cfg["num_workers"],
                             pin_memory=cfg["pin_memory"])

    model = ResNet1D(in_channels=cfg["in_channels"],
                     num_classes=cfg["num_classes"],
                     base_filters=cfg["base_filters"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_fids = [], []
    for signals, fids in test_loader:
        logits = model(signals.to(device, non_blocking=True))
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_fids.extend(fids.numpy())

    return (pd.DataFrame({"Id": all_fids, "Label": all_preds})
            .sort_values("Id").reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix formatting helper
# ══════════════════════════════════════════════════════════════════════════════

def format_confusion_matrix(all_labels, all_preds) -> str:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(N_CLASSES)))
    header  = "Pred→  " + "  ".join(f"  L{c}" for c in range(N_CLASSES))
    divider = "─" * len(header)
    rows    = [header, divider]
    for i, row in enumerate(cm):
        rows.append(f"True L{i} | " + " | ".join(f"{v:5d}" for v in row))
    rows.append("(Rows = True Label, Columns = Predicted Label)")
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HAR 1D-ResNet Training Pipeline v6")
    parser.add_argument("--all-folds",  action="store_true")
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

    # Re-apply seed after any CLI arg mutations (belt-and-suspenders)
    seed_everything(CFG["seed"])

    # ── Logger setup ──────────────────────────────────────────────────────────
    logger = DualLogger(CFG["log_path"])

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header_lines = [
        "=" * 70,
        "  HAR 1D-ResNet Training Pipeline  (v6)",
        "=" * 70,
        f"  Log file    : {os.path.abspath(CFG['log_path'])}",
        f"  Device      : {device}",
    ]
    if device.type == "cuda":
        header_lines += [
            f"  GPU         : {torch.cuda.get_device_name(0)}",
            f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        ]
    header_lines += [
        f"  Seed        : {CFG['seed']}  (deterministic=True, benchmark=False)",
        f"  In channels : {CFG['in_channels']}  (4 pruned: mean_x/y/z + std_mag)",
        f"  Epochs      : {CFG['epochs']}",
        f"  Batch       : {CFG['batch_size']}",
        f"  LR          : {CFG['lr']}",
        f"  Weight Decay: {CFG['weight_decay']}",
        f"  Focal γ     : {CFG['focal_gamma']}  (uniform, all classes)",
        f"  Label Smooth: ε={CFG['label_smoothing']}",
        f"  Mixup α     : {CFG['mixup_alpha']}  (Beta distribution, train only)",
        f"  Patience    : {CFG['patience']}",
        f"  Folds       : {'all 5' if CFG['run_all_folds'] else 'fold 1 only'}",
        f"  Ensemble    : F1-Weighted TTA (×1.00 / ×1.05 / ×0.95)",
        "=" * 70,
    ]
    for line in header_lines:
        logger.log(line)

    # ── Run folds ─────────────────────────────────────────────────────────────
    folds_to_run    = list(range(N_FOLDS)) if CFG["run_all_folds"] else [0]
    fold_results    = []   # (fold_idx, best_f1, norm_params, ckpt_path, report)
    all_fold_preds  = []
    all_fold_labels = []

    t_total_start = time.time()

    for fold_idx in folds_to_run:
        best_f1, norm_params, ckpt_path, (val_preds, val_labels), report = train_fold(
            fold_idx, device, CFG, logger
        )
        fold_results.append((fold_idx, best_f1, norm_params, ckpt_path, report))
        if val_preds is not None:
            all_fold_preds.extend(val_preds)
            all_fold_labels.extend(val_labels)

    t_total = time.time() - t_total_start

    # ── Cross-validation summary ──────────────────────────────────────────────
    logger.log("\n" + "=" * 70)
    logger.log("CROSS-VALIDATION SUMMARY")
    logger.log("=" * 70)

    f1_scores = [r[1] for r in fold_results]
    for fi, best_f1, _, _, _ in fold_results:
        logger.log(f"  Fold {fi+1}:  Best Val Macro F1 = {best_f1:.6f}")
    logger.log(f"\n  Mean F1 = {np.mean(f1_scores):.6f}  ±  {np.std(f1_scores):.6f}")
    logger.log(f"  Total wall-clock time: {t_total / 60:.1f} min")

    # ── Per-fold classification reports → file only ───────────────────────────
    if len(fold_results) > 0:
        logger.log_file("\n--- Per-Fold Classification Reports (Best Epoch) ---")
        for fi, best_f1, _, _, report in fold_results:
            logger.log_file(f"\n[Fold {fi+1}]  Val Macro F1 = {best_f1:.6f}")
            logger.log_file(report)

    # ── Aggregate confusion matrix → file only ────────────────────────────────
    if len(all_fold_labels) > 0:
        cm_text = format_confusion_matrix(all_fold_labels, all_fold_preds)
        logger.log_file("\n--- Aggregate Confusion Matrix (all CV validation folds) ---")
        logger.log_file(cm_text)
        # Also surface summary to console
        logger.log("\n--- Aggregate Confusion Matrix (all CV validation folds) ---")
        logger.log(cm_text)

    # ══════════════════════════════════════════════════════════════════════════
    # F1-Weighted TTA Ensemble Inference  (Modules 4 + 5)
    # ══════════════════════════════════════════════════════════════════════════
    # This block executes automatically after the CV loop.
    # For each fold it:
    #   - Loads the best checkpoint (fold_N_best.pth)
    #   - Runs TTA inference (X, X×1.05, X×0.95) and averages the three softmax
    #     distributions to produce a single fold probability vector
    # Then it combines per-fold vectors using F1-weighted averaging:
    #   P_ensemble = Σ(F1_i × P_i) / Σ(F1_i)
    # and writes the final prediction map to submission_v5_ensemble.csv.
    # NOTE: Mixup is strictly NOT applied during inference.
    # ──────────────────────────────────────────────────────────────────────────
    if len(fold_results) > 0:
        ensemble_submission = ensemble_inference(
            fold_results=fold_results,
            cfg=CFG,
            device=device,
            logger=logger,
        )

    logger.log("\n" + "=" * 70)
    logger.log("  Pipeline v6 complete.")
    logger.log(f"  Ensemble submission ready : {CFG['submission_path']}")
    logger.log("=" * 70)

    logger.close()


if __name__ == "__main__":
    main()
