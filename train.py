"""
train.py
--------
Hierarchical Two-Stage Training Pipeline for HAR 1D-ResNet (v7).

Architecture
────────────
  Stage 1 (Generalist)  : 5-class ResNet1D  – classifies L0, L1_2_Merged, L3, L4, L5
  Stage 2 (Specialist)  : Binary  ResNet1D  – distinguishes L1 vs L2 among merged samples

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
    • 5-fold CV on L1/L2 samples only
    • BCEWithLogitsLoss with computed pos_weight = count(L1) / count(L2)
    • No Mixup (specialist binary task)
    • AdamW + CosineAnnealingLR
    • Early stop on validation Macro F1 (patience=25)
    • Saves: checkpoints/stage2_fold_{n}_best.pth

Inference (Hierarchical Routing)
─────────────────────────────────
  1. Run all test samples through Stage 1 ensemble (5 folds × TTA ×3 scales)
     → F1-weighted softmax averaging → argmax → 5-class prediction
  2. If predicted class == 1 (L1_2_Merged) → route to Stage 2
     Else → accept Stage 1 prediction; map back to original label space
  3. Stage 2 specialist runs routed samples through 5 folds × TTA × Sigmoid
     → F1-weighted probability averaging → threshold 0.5 → L1 or L2
  4. Recombine and write: submission_v7_hierarchical.csv

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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

from dataset import (
    build_fold_datasets, build_test_dataset,
    load_all_samples, get_fold_splits,
    N_FOLDS, N_CLASSES, N_CLASSES_S1, N_CLASSES_S2,
    STAGE1_MAP, STAGE2_MAP,
)
from model import ResNet1D

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
    "submission_path"   : "./submission_v7_hierarchical.csv",
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

    # Model ───────────────────────────────────────────────────────────────────
    "in_channels"       : 8,           # v7: 8 stable channels
    "base_filters"      : 64,          # v7: restored full width

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
    # Guard against zero counts (should not happen, but just in case)
    if count_pos < 1.0:
        count_pos = 1.0
    pos_weight = torch.tensor([count_neg / count_pos],
                               dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch_s2(model, loader, criterion, optimizer, device) -> float:
    """
    Train Stage 2 for one epoch with BCEWithLogitsLoss.
    No Mixup – binary specialist task requires clean decision boundaries.
    Model outputs 1 logit per sample (squeeze applied before loss).
    """
    model.train()
    running_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)          # (B, 8, 300)
        labels  = labels.to(device,  non_blocking=True).float()  # (B,) float

        optimizer.zero_grad()
        logits = model(signals).squeeze(-1)   # (B, 1) → (B,)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * signals.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_s2(model, loader, criterion, device, threshold: float = 0.5):
    """Validation for Stage 2 (binary, BCEWithLogitsLoss)."""
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
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase A] --- Fold {fold_num}/{N_FOLDS} Stage1 Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE A | FOLD {fold_num} / {N_FOLDS}  – Stage 1 Generalist (5-class)")
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

    model_s1 = ResNet1D(in_channels=cfg["in_channels"],
                        num_classes=cfg["num_classes_s1"],
                        base_filters=cfg["base_filters"]).to(device)
    n_params = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    logger.log_file(f"  Model_Stage1: ResNet1D  |  Params: {n_params:,}")

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
# Phase B: Per-fold Stage 2 training
# ══════════════════════════════════════════════════════════════════════════════

def train_fold_s2(fold_idx: int, device: torch.device, cfg: dict,
                  logger: DualLogger):
    """
    Train one CV fold for Stage 2 (2-class specialist: L1 vs L2).
    Model uses 1 output logit with BCEWithLogitsLoss.

    Returns
    -------
    best_val_f1     : float
    checkpoint_path : str
    best_val_report : str
    """
    fold_num = fold_idx + 1

    logger.log(f"\n[Phase B] --- Fold {fold_num}/{N_FOLDS} Stage2 Starting ---")
    logger.log_file("=" * 70)
    logger.log_file(f"PHASE B | FOLD {fold_num} / {N_FOLDS}  – Stage 2 Specialist (L1 vs L2)")
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
        return 0.0, ckpt_path, ""

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])

    # Stage 2 model: 1 output logit for BCEWithLogitsLoss
    model_s2 = ResNet1D(in_channels=cfg["in_channels"],
                        num_classes=1,
                        base_filters=cfg["base_filters"]).to(device)
    n_params = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    logger.log_file(f"  Model_Stage2: ResNet1D (1 logit)  |  Params: {n_params:,}")

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
        f"--- [S2] Fold {fold_num} done | "
        f"Best Val Macro F1: {early_stop.best_score:.6f} "
        f"(epoch {early_stop.best_epoch}) ---"
    )
    logger.log(fold_summary)
    logger.log_file("")
    logger.log_file(f"  [S2] Classification Report @ Best Epoch ({early_stop.best_epoch}):")
    logger.log_file(best_val_report)
    logger.log_file("─" * 70)

    return early_stop.best_score, ckpt_path, best_val_report


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
# Module 4: Hierarchical Softmax Averaging Ensemble Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def hierarchical_inference(s1_fold_results: list,
                            s2_fold_results: list,
                            cfg:             dict,
                            device:          torch.device,
                            logger:          DualLogger) -> pd.DataFrame:
    """
    Full hierarchical routing inference pipeline.

    Step 1: Stage 1 TTA ensemble (5 folds × 3 TTA scales, F1-weighted avg)
    Step 2: argmax → accept non-merged predictions, flag merged (idx==1)
    Step 3: Stage 2 specialist TTA ensemble on flagged samples
    Step 4: Recombine and write submission_v7_hierarchical.csv

    Parameters
    ----------
    s1_fold_results : list of (fold_idx, best_f1, norm_params, ckpt_path, report)
    s2_fold_results : list of (fold_idx, best_f1, ckpt_path, report)
    cfg             : dict
    device          : torch.device
    logger          : DualLogger

    Returns
    -------
    submission : pd.DataFrame  with columns ['Id', 'Label']
    """
    logger.log("\n" + "=" * 70)
    logger.log("  HIERARCHICAL INFERENCE  (Stage1 → Route → Stage2)")
    logger.log("=" * 70)

    # ── Load test data using norm params from Stage 1 fold 1 ─────────────────
    first_norm   = s1_fold_results[0][2]   # (mean, std)
    test_ds, file_ids = build_test_dataset(
        cfg["test_root"], first_norm[0], first_norm[1])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    n_test = len(test_ds)
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

        model_s1 = ResNet1D(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes_s1"],
            base_filters=cfg["base_filters"],
        ).to(device)
        model_s1.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_s1.eval()

        fold_probs = []
        for signals, _ in test_loader:
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
    # STEP 2 + 3: Stage 2 Specialist for routed samples
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
        logger.log("\n  [Step 3] Stage 2 Specialist Ensemble for routed samples ...")

        # Extract routed test signals (still normalised, on CPU via dataset)
        # Re-load test signals tensor by iterating test_loader in order
        all_test_signals = []
        for signals, _ in test_loader:
            all_test_signals.append(signals)
        all_test_signals = torch.cat(all_test_signals, dim=0)   # (N, 8, 300)

        routed_signals = all_test_signals[s2_route_mask]        # (n_routed, 8, 300)

        # Build a simple tensor dataset for the routed samples
        from torch.utils.data import TensorDataset
        routed_ds     = TensorDataset(routed_signals)
        routed_loader = DataLoader(
            routed_ds,
            batch_size=cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=0,
        )

        s2_weighted_prob_sum = np.zeros(n_routed, dtype=np.float64)
        s2_f1_total          = 0.0

        for fold_idx, best_f1, ckpt_path, _ in s2_fold_results:
            fold_num = fold_idx + 1
            if best_f1 <= 0.0 or not os.path.isfile(ckpt_path):
                logger.log(f"    S2 Fold {fold_num}: checkpoint missing / F1=0 – skipping.")
                continue
            logger.log(f"    S2 Fold {fold_num}: loading {ckpt_path}  (Val F1={best_f1:.6f})")

            model_s2 = ResNet1D(
                in_channels=cfg["in_channels"],
                num_classes=1,
                base_filters=cfg["base_filters"],
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

            fold_probs_s2_arr     = np.concatenate(fold_probs_s2, axis=0)   # (n_routed,)
            s2_weighted_prob_sum += best_f1 * fold_probs_s2_arr
            s2_f1_total          += best_f1
            logger.log(f"      → accumulated (weight={best_f1:.6f})")

        if s2_f1_total > 0.0:
            s2_ensemble_probs = s2_weighted_prob_sum / s2_f1_total   # (n_routed,) prob of L2
            # Apply decision threshold: >= threshold → L2 (orig 2), else → L1 (orig 1)
            s2_preds_binary = (s2_ensemble_probs >= cfg["s2_threshold"]).astype(np.int64)
            # Binary 0 → original L1, binary 1 → original L2
            s2_preds_orig   = np.where(s2_preds_binary == 0, 1, 2)
        else:
            # Fallback: default all routed samples to L1
            logger.log("  WARNING: No valid S2 folds – defaulting routed samples to L1.")
            s2_preds_orig = np.ones(n_routed, dtype=np.int64)

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
        description="HAR 1D-ResNet v7 – Hierarchical Two-Stage Training Pipeline")
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

    # ── Header ────────────────────────────────────────────────────────────────
    header_lines = [
        "=" * 70,
        "  HAR 1D-ResNet v7 – Hierarchical Two-Stage Classifier",
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
        f"  Base filters  : {CFG['base_filters']}  (Stem=64, S1=64, S2=128, S3=256, S4=512)",
        f"  S1 classes    : {CFG['num_classes_s1']}  (L0 | L1_2_Merged | L3 | L4 | L5)",
        f"  S2 classes    : 1 logit / BCE  (L1 vs L2 specialist)",
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
        f"  Ensemble      : F1-Weighted TTA (×1.00 / ×1.05 / ×0.95)",
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
    logger.log("  PHASE A – Stage 1 Generalist Training (5-class)")
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
    # PHASE B – Train Stage 2 Specialist
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n" + "═" * 70)
    logger.log("  PHASE B – Stage 2 Specialist Training (L1 vs L2 binary)")
    logger.log("═" * 70)

    t_phaseB_start  = time.time()
    s2_fold_results = []   # (fold_idx, best_f1, ckpt_path, report)
    s2_all_preds    = []
    s2_all_labels   = []

    for fold_idx in folds_to_run:
        best_f1, ckpt_path, report = train_fold_s2(
            fold_idx, device, CFG, logger)
        s2_fold_results.append((fold_idx, best_f1, ckpt_path, report))

    t_phaseB = time.time() - t_phaseB_start

    logger.log("\n" + "═" * 70)
    logger.log("  PHASE B SUMMARY – Stage 2 Cross-Validation")
    logger.log("═" * 70)
    s2_f1_scores = [r[1] for r in s2_fold_results]
    for fi, bf1, _, _ in s2_fold_results:
        logger.log(f"  [S2] Fold {fi+1}:  Best Val Macro F1 = {bf1:.6f}")
    if len(s2_f1_scores) > 0:
        logger.log(f"\n  [S2] Mean F1 = {np.mean(s2_f1_scores):.6f}  ±  {np.std(s2_f1_scores):.6f}")
    logger.log(f"  Phase B wall-clock: {t_phaseB / 60:.1f} min")
    logger.log(f"  Total training wall-clock: {(t_phaseA + t_phaseB) / 60:.1f} min")

    # ════════════════════════════════════════════════════════════════════════
    # INFERENCE – Hierarchical Routing
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
    logger.log("  Pipeline v7 complete.")
    logger.log(f"  Hierarchical submission: {CFG['submission_path']}")
    logger.log("=" * 70)

    logger.close()


if __name__ == "__main__":
    main()
