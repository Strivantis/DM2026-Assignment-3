"""
experiment_preprocessing.py
----------------------------
Phase A Preprocessing Independent Quantification Experiment — v13 Pipeline.

Experiment: Single-fold (Fold 1 only), 80-epoch runs isolating each preprocessing
component to measure its individual contribution to Validation Macro F1.

Four sequential variations are executed:
  Variation 1 – Baseline (Full v13 preprocessing):
    • Global Z-score normalisation  ✓
    • Batch Mixup (α=0.2)           ✓
    • Augmentation (AUG_PROB=0.5)   ✓

  Variation 2 – No Mixup:
    • Global Z-score normalisation  ✓
    • Batch Mixup                   ✗  (bypassed: lam=1.0, no interpolation)
    • Augmentation                  ✓

  Variation 3 – No Augmentation:
    • Global Z-score normalisation  ✓
    • Batch Mixup                   ✓
    • Augmentation                  ✗  (AUG_PROB forced to 0.0)

  Variation 4 – No Z-Score:
    • Global Z-score normalisation  ✗  (raw unnormalised signals passed to model)
    • Batch Mixup                   ✓
    • Augmentation                  ✓

Metric: Validation Set Macro F1 at the best epoch of Fold 1 (patience-limited,
        but capped at exactly 80 epochs).

Usage:
    python experiment_preprocessing.py

Outputs: prints final Validation Macro F1 for Fold 1 after each variant finishes.
"""

import os
import sys
import time
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from dataset import (
    load_all_samples,
    normalize,
    compute_normalization_params,
    HARDataset,
    N_FOLDS,
    N_CLASSES,
    AUG_PROB,
)
from model import InceptionTime1D

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — v13 configuration (mirrored from train.py CFG)
# ──────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 6
CLASS_NAMES = ["L0", "L1", "L2", "L3", "L4", "L5"]

CFG = {
    "train_root"     : "./train/train",
    "checkpoint_dir" : "./checkpoints",
    "epochs"         : 80,
    "batch_size"     : 128,
    "lr"             : 3e-4,
    "weight_decay"   : 1e-3,
    "num_workers"    : 4,
    "pin_memory"     : True,
    "T_max"          : 80,
    "eta_min"        : 1e-6,
    "patience"       : 25,
    "in_channels"    : 8,
    "nb_filters"     : 64,
    "bottleneck"     : 32,
    "depth"          : 6,
    "se_reduction"   : 16,
    "num_classes"    : NUM_CLASSES,
    "focal_gamma"    : 2.0,
    "label_smoothing": 0.1,
    "mixup_alpha"    : 0.2,
    "seed"           : 42,
    "fold_idx"       : 0,          # Fold 1 (0-indexed)
}

# Experiment variation definitions
VARIATIONS = [
    {
        "id"              : 1,
        "name"            : "Variation 1 – Baseline (Full v13 preprocessing)",
        "use_mixup"       : True,
        "aug_prob"        : AUG_PROB,   # 0.5 from dataset.py
        "use_zscore_norm" : True,
    },
    {
        "id"              : 2,
        "name"            : "Variation 2 – No Mixup",
        "use_mixup"       : False,
        "aug_prob"        : AUG_PROB,
        "use_zscore_norm" : True,
    },
    {
        "id"              : 3,
        "name"            : "Variation 3 – No Augmentation (AUG_PROB=0.0)",
        "use_mixup"       : True,
        "aug_prob"        : 0.0,
        "use_zscore_norm" : True,
    },
    {
        "id"              : 4,
        "name"            : "Variation 4 – No Z-Score Normalisation",
        "use_mixup"       : True,
        "aug_prob"        : AUG_PROB,
        "use_zscore_norm" : False,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Lock all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss (identical to train.py)
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Alpha-weighted Focal Loss with uniform γ and label smoothing ε."""

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
                     num_classes: int,
                     device: torch.device,
                     gamma: float = 2.0,
                     label_smoothing: float = 0.1) -> FocalLoss:
    counts  = class_counts.astype(np.float32)
    total   = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.mean()
    alpha   = torch.tensor(weights, dtype=torch.float32, device=device)
    return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)


# ──────────────────────────────────────────────────────────────────────────────
# Mixup augmentation (identical to train.py)
# ──────────────────────────────────────────────────────────────────────────────

def mixup_batch(signals:         torch.Tensor,
                labels:          torch.Tensor,
                alpha:           float,
                num_classes:     int,
                label_smoothing: float,
                device:          torch.device):
    """Batch-level Mixup. Returns x_mixed, y_mixed, lam."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation-override HARDataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

import dataset as _dataset_module


class HARDatasetAugOverride(torch.utils.data.Dataset):
    """
    Wraps the standard HARDataset but overrides the global AUG_PROB constant
    at __getitem__ time to support Variation 3 (No Augmentation).

    Parameters
    ----------
    signals     : (N, 300, 8) normalised signal array
    labels      : (N,) integer labels
    orig_labels : (N,) original 0-5 labels (for augmentation protection)
    is_train    : bool
    aug_prob    : float – override for AUG_PROB (0.0 disables all augmentation)
    """

    def __init__(self,
                 signals:     np.ndarray,
                 labels:      np.ndarray,
                 orig_labels: np.ndarray,
                 is_train:    bool  = False,
                 aug_prob:    float = 0.5):
        self.signals     = signals.astype(np.float32)
        self.labels      = labels.astype(np.int64)
        self.orig_labels = orig_labels.astype(np.int64)
        self.is_train    = is_train
        self.aug_prob    = aug_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal     = self.signals[idx].copy()   # (300, 8)
        label      = self.labels[idx]
        orig_label = self.orig_labels[idx]

        if self.is_train:
            # Temporarily patch the module-level AUG_PROB used by apply_augmentation
            original_aug_prob = _dataset_module.AUG_PROB
            _dataset_module.AUG_PROB = self.aug_prob
            try:
                signal = _dataset_module.apply_augmentation(signal, int(orig_label))
            finally:
                _dataset_module.AUG_PROB = original_aug_prob

        # Transpose: (300, 8) → (8, 300)
        signal = signal.T
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping (identical logic to train.py)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    mixup_alpha: float, label_smoothing: float,
                    num_classes: int, use_mixup: bool) -> float:
    """Train SE-InceptionTime for one epoch.

    use_mixup=False → passes raw hard labels through without interpolation.
    """
    model.train()
    running_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        if use_mixup:
            x_in, y_in, _ = mixup_batch(
                signals, labels,
                alpha=mixup_alpha,
                num_classes=num_classes,
                label_smoothing=label_smoothing,
                device=device,
            )
        else:
            # No mixup: use original signals and integer labels directly
            x_in = signals
            y_in = labels

        optimizer.zero_grad()
        logits = model(x_in)
        loss   = criterion(logits, y_in)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * signals.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Validation pass. Returns (avg_loss, accuracy, macro_f1, preds, labels, probs)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)

        logits = model(signals)
        running_loss += criterion(logits, labels).item() * signals.size(0)

        probs = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)

    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels, all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Core: run one preprocessing variation for Fold 1
# ──────────────────────────────────────────────────────────────────────────────

def run_variation(variation: dict,
                  all_signals: np.ndarray,
                  all_labels: np.ndarray,
                  all_groups: np.ndarray,
                  global_mean: np.ndarray,
                  global_std: np.ndarray,
                  device: torch.device,
                  cfg: dict) -> dict:
    """
    Train SE-InceptionTime on Fold 1 for exactly 80 epochs under a single
    preprocessing variation, then return the best validation Macro F1.

    Parameters
    ----------
    variation      : dict describing the preprocessing switches
    all_signals    : (N, 300, 8)  raw (unnormalised) signal array
    all_labels     : (N,) integer class labels
    all_groups     : (N,) integer user group IDs
    global_mean    : (8,) global training-set mean
    global_std     : (8,) global training-set std
    device         : torch.device
    cfg            : pipeline config dict

    Returns
    -------
    dict with keys: best_val_f1, best_epoch, history, classification_report_str
    """
    from dataset import get_fold_splits

    var_id        = variation["id"]
    var_name      = variation["name"]
    use_mixup     = variation["use_mixup"]
    aug_prob      = variation["aug_prob"]
    use_zscore    = variation["use_zscore_norm"]
    fold_idx      = cfg["fold_idx"]   # 0 → Fold 1
    fold_num      = fold_idx + 1

    print(f"\n{'─'*70}")
    print(f"  [{var_id}/4] {var_name}")
    print(f"  Fold {fold_num} | Epochs: {cfg['epochs']} | "
          f"use_mixup={use_mixup}  aug_prob={aug_prob}  "
          f"use_zscore={use_zscore}")
    print(f"{'─'*70}")

    # ── Seed per variation to ensure fair comparisons ─────────────────────────
    seed_everything(cfg["seed"])

    # ── Get fold split indices ─────────────────────────────────────────────────
    train_idx, val_idx = get_fold_splits(
        all_signals, all_labels, all_groups, fold_idx)

    # ── Apply (or skip) Z-score normalisation ─────────────────────────────────
    if use_zscore:
        train_signals = (all_signals[train_idx] - global_mean[np.newaxis, np.newaxis, :]) \
                        / global_std[np.newaxis, np.newaxis, :]
        val_signals   = (all_signals[val_idx]   - global_mean[np.newaxis, np.newaxis, :]) \
                        / global_std[np.newaxis, np.newaxis, :]
        train_signals = train_signals.astype(np.float32)
        val_signals   = val_signals.astype(np.float32)
    else:
        # No normalisation: pass raw signal values to the model
        train_signals = all_signals[train_idx].astype(np.float32)
        val_signals   = all_signals[val_idx].astype(np.float32)

    train_labels = all_labels[train_idx]
    val_labels   = all_labels[val_idx]

    print(f"  Train: {len(train_idx)} samples  Val: {len(val_idx)} samples")
    print(f"  Class counts (train): {np.bincount(train_labels, minlength=NUM_CLASSES).tolist()}")

    # ── Build datasets (augmentation-override wrapper) ─────────────────────────
    # Validation set: always no augmentation (is_train=False)
    train_ds = HARDatasetAugOverride(
        train_signals, train_labels, train_labels,
        is_train=True, aug_prob=aug_prob)
    val_ds = HARDatasetAugOverride(
        val_signals, val_labels, val_labels,
        is_train=False, aug_prob=0.0)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

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
    print(f"  Model: SE-InceptionTime1D  Params: {n_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    criterion = build_focal_loss(
        class_counts, cfg["num_classes"], device,
        gamma=cfg["focal_gamma"],
        label_smoothing=cfg["label_smoothing"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_name = f"exp_preproc_var{var_id}_fold{fold_num}.pth"
    ckpt_path = os.path.join(cfg["checkpoint_dir"], ckpt_name)
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    best_val_report = ""
    best_val_preds  = None
    best_val_labels = None
    history         = []

    # ── Training loop (exactly 80 epochs, early stopping permitted) ───────────
    for epoch in range(1, cfg["epochs"] + 1):
        t_start    = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=cfg["mixup_alpha"],
            label_smoothing=cfg["label_smoothing"],
            num_classes=cfg["num_classes"],
            use_mixup=use_mixup,
        )
        scheduler.step()

        val_loss, val_acc, val_f1, val_preds, val_lbs, val_probs = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - t_start
        lr_now  = scheduler.get_last_lr()[0]

        print(f"  Ep {epoch:>3}/{cfg['epochs']} | "
              f"LR: {lr_now:.1e} | "
              f"Loss T:{train_loss:.4f} V:{val_loss:.4f} | "
              f"Val Acc: {val_acc:5.2f}% | "
              f"Val F1: {val_f1:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        if val_f1 >= early_stop.best_score:
            best_val_report = classification_report(
                val_lbs, val_preds,
                target_names=CLASS_NAMES, digits=4, zero_division=0)
            best_val_preds  = val_preds.copy()
            best_val_labels = val_lbs.copy()

        should_stop = early_stop.step(val_f1, model, epoch)
        if should_stop:
            print(f"  >> Early stop @ epoch {epoch}.  "
                  f"Best epoch: {early_stop.best_epoch}  "
                  f"Best F1: {early_stop.best_score:.4f}")
            break

    print(f"\n  ✓ {var_name}")
    print(f"    Best Val Macro F1 : {early_stop.best_score:.4f}  "
          f"(epoch {early_stop.best_epoch})")
    print(f"    Checkpoint saved  : {ckpt_path}")

    return {
        "variation_id"     : var_id,
        "variation_name"   : var_name,
        "best_val_f1"      : early_stop.best_score,
        "best_epoch"       : early_stop.best_epoch,
        "history"          : history,
        "classification_report": best_val_report,
        "best_val_preds"   : best_val_preds,
        "best_val_labels"  : best_val_labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    seed_everything(CFG["seed"])

    print("=" * 70)
    print("  Phase A Preprocessing Quantification — v13 SE-InceptionTime")
    print("  Fold 1 Only | 80 Epochs | 4 Isolated Preprocessing Variations")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Load ALL training samples ─────────────────────────────────────
    TRAIN_ROOT = CFG["train_root"]
    print(f"\n[Data] Loading training samples from '{TRAIN_ROOT}' ...")
    t0 = time.time()
    all_signals, all_labels, all_groups, _ = load_all_samples(TRAIN_ROOT)
    print(f"[Data] Loaded {len(all_signals)} samples in {time.time()-t0:.1f}s")
    print(f"[Data] Signal shape : {all_signals.shape}")
    print(f"[Data] Label counts : {np.bincount(all_labels, minlength=NUM_CLASSES).tolist()}")

    # ── Step 2: Compute GLOBAL normalisation parameters ────────────────────────
    # These are computed from the full training set as in v13.
    # For Variation 4 (No Z-Score), raw unnormalised signals are used instead.
    print("[Data] Computing global Z-score normalisation parameters ...")
    global_mean, global_std = compute_normalization_params(all_signals)
    print(f"[Data] Global mean : {np.round(global_mean, 4).tolist()}")
    print(f"[Data] Global std  : {np.round(global_std,  4).tolist()}")

    # ── Step 3: Run all 4 variations sequentially ──────────────────────────────
    results       = []
    t_total_start = time.time()

    for variation in VARIATIONS:
        t_var   = time.time()
        result  = run_variation(
            variation   = variation,
            all_signals = all_signals,
            all_labels  = all_labels,
            all_groups  = all_groups,
            global_mean = global_mean,
            global_std  = global_std,
            device      = device,
            cfg         = CFG,
        )
        result["elapsed"] = time.time() - t_var
        results.append(result)

        print(f"\n  [Variation {variation['id']}/4 complete]  "
              f"Best Val Macro F1 = {result['best_val_f1']:.4f}  "
              f"({result['elapsed']:.1f}s)\n")

    total_elapsed = time.time() - t_total_start

    # ── Step 4: Print final summary ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  PREPROCESSING QUANTIFICATION RESULTS — Fold 1, 80 Epochs")
    print("=" * 70)
    print(f"  {'Variation':<55} {'Best Macro F1':>14} {'Best Epoch':>11}")
    print("  " + "─" * 84)

    for r in results:
        print(f"  {r['variation_name']:<55} "
              f"{r['best_val_f1']:>14.4f} "
              f"{r['best_epoch']:>11d}")

    # Delta analysis relative to Variation 1 (Baseline)
    baseline_f1 = results[0]["best_val_f1"]
    print(f"\n  DELTA vs Variation 1 Baseline (Full v13 preprocessing):")
    print(f"  {'Variation':<55} {'ΔMacro F1':>14}")
    print("  " + "─" * 72)
    for r in results:
        delta = r["best_val_f1"] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        print(f"  {r['variation_name']:<55} {sign}{delta:>13.4f}")

    print(f"\n  Total wall-clock: {total_elapsed/60:.1f} min")
    print("=" * 70)

    # ── Step 5: Print per-variation classification reports ────────────────────
    print("\n\n  CLASSIFICATION REPORTS (Best Epoch for each Variation)")
    for r in results:
        print(f"\n  {'─'*70}")
        print(f"  {r['variation_name']}  (Best Epoch: {r['best_epoch']}  "
              f"Val Macro F1: {r['best_val_f1']:.4f})")
        print(f"  {'─'*70}")
        if r["classification_report"]:
            for line in r["classification_report"].splitlines():
                print(f"  {line}")
        else:
            print("  (no report available)")

    print("\n" + "=" * 70)
    print("  Experiment complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
