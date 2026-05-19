"""
train.py
--------
End-to-end training pipeline for the HAR 1D-ResNet.

Features
────────
  • 5-fold StratifiedGroupKFold cross-validation
  • Class-Weighted CrossEntropyLoss  (counters severe class imbalance)
  • AdamW optimiser + CosineAnnealingLR scheduler
  • Early stopping monitored on Validation Macro F1-Score
  • Dual-stream logging: compressed single-line console + full-detail file log
  • Best checkpoint saved per fold
  • Full classification report + confusion matrix at final CV summary
  • Final predictions on the held-out test set → submission.csv

Usage
─────
    conda activate PyTorch
    python train.py               # fold 0 only
    python train.py --all-folds   # full 5-fold CV

DO NOT run this script automatically – see task constraints.
"""

import os
import sys
import time
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
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Paths ───────────────────────────────────────────────────────────────────
    "train_root"      : "./train/train",
    "test_root"       : "./test/test",
    "submission_path" : "./submission.csv",
    "checkpoint_dir"  : "./checkpoints",
    "log_path"        : "./training_log.txt",

    # Training ────────────────────────────────────────────────────────────────
    "epochs"          : 80,
    "batch_size"      : 128,
    "lr"              : 3e-4,
    "weight_decay"    : 1e-4,
    "num_workers"     : 4,
    "pin_memory"      : True,

    # CosineAnnealingLR
    "T_max"           : 80,
    "eta_min"         : 1e-6,

    # Early stopping ──────────────────────────────────────────────────────────
    "patience"        : 15,

    # Model ───────────────────────────────────────────────────────────────────
    "in_channels"     : 6,
    "num_classes"     : 6,
    "base_filters"    : 64,

    # Cross-validation ────────────────────────────────────────────────────────
    "run_all_folds"   : False,
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
    log_console(msg) – write to CONSOLE only  (transient status that is too noisy for file)
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
# Loss function
# ══════════════════════════════════════════════════════════════════════════════

def build_class_weighted_loss(class_counts: np.ndarray,
                              device: torch.device) -> nn.CrossEntropyLoss:
    """
    Compute inverse-frequency class weights.
    weight_c = total / (N_CLASSES × count_c),  then normalised so mean=1.
    """
    counts  = class_counts.astype(np.float32)
    total   = counts.sum()
    weights = total / (N_CLASSES * counts)
    weights = weights / weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


# ══════════════════════════════════════════════════════════════════════════════
# Early stopping helper
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4,
                 checkpoint_path: str = "best_model.pt"):
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

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(signals), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * signals.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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

    # ── Console: clean single-line separator ──────────────────────────────────
    logger.log(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} Starting ---")

    # ── File: verbose fold header ─────────────────────────────────────────────
    logger.log_file("=" * 70)
    logger.log_file(f"FOLD {fold_idx + 1} / {N_FOLDS}  ──  Starting")
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

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_class_weighted_loss(class_counts, device)
    counts  = class_counts.astype(np.float32)
    weights = counts.sum() / (N_CLASSES * counts)
    weights = weights / weights.mean()
    logger.log_file("  Loss: Class-Weighted CrossEntropyLoss")
    logger.log_file("  Effective class weights: " +
                    "  ".join(f"L{c}:{w:.3f}" for c, w in enumerate(weights)))

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["T_max"], eta_min=cfg["eta_min"])

    # ── Early stopping ────────────────────────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path  = os.path.join(cfg["checkpoint_dir"], f"fold{fold_idx}_best.pt")
    early_stop = EarlyStopping(patience=cfg["patience"], checkpoint_path=ckpt_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_preds  = (None, None)
    best_val_report = ""

    for epoch in range(1, cfg["epochs"] + 1):
        t_start    = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - t_start
        lr_now  = scheduler.get_last_lr()[0]

        # ── Console: single compact line ──────────────────────────────────────
        star = " *" if val_f1 >= early_stop.best_score else "  "
        logger.log_console(
            f"[Fold {fold_idx+1}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.1e} | "
            f"Loss: T:{train_loss:.4f} / V:{val_loss:.4f} | "
            f"Val Acc: {val_acc:6.2f}% | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s{star}"
        )

        # ── File: same line PLUS extra detail on next lines ───────────────────
        logger.log_file(
            f"[Fold {fold_idx+1}/{N_FOLDS}] Ep {epoch:>3}/{cfg['epochs']} | "
            f"LR: {lr_now:.1e} | "
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
        f"--- Fold {fold_idx+1} done | "
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
# Test-set inference
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
    parser = argparse.ArgumentParser(description="HAR 1D-ResNet Training Pipeline")
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

    # ── Logger setup ──────────────────────────────────────────────────────────
    logger = DualLogger(CFG["log_path"])

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header_lines = [
        "=" * 70,
        "  HAR 1D-ResNet Training Pipeline",
        "=" * 70,
        f"  Log file : {os.path.abspath(CFG['log_path'])}",
        f"  Device   : {device}",
    ]
    if device.type == "cuda":
        header_lines += [
            f"  GPU      : {torch.cuda.get_device_name(0)}",
            f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        ]
    header_lines += [
        f"  Epochs   : {CFG['epochs']}",
        f"  Batch    : {CFG['batch_size']}",
        f"  LR       : {CFG['lr']}",
        f"  Patience : {CFG['patience']}",
        f"  Folds    : {'all 5' if CFG['run_all_folds'] else 'fold 0 only'}",
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

    # ── Per-fold classification reports → console + file at summary stage ─────
    if len(fold_results) > 0:
        logger.log("\n--- Per-Fold Classification Reports (Best Epoch) ---")
        for fi, best_f1, _, _, report in fold_results:
            logger.log(f"\n[Fold {fi+1}]  Val Macro F1 = {best_f1:.6f}")
            logger.log(report)

    # ── Aggregate confusion matrix ────────────────────────────────────────────
    if len(all_fold_labels) > 0:
        logger.log("\n--- Aggregate Confusion Matrix (all CV validation folds) ---")
        cm_text = format_confusion_matrix(all_fold_labels, all_fold_preds)
        logger.log(cm_text)

    # ── Best fold for test inference ──────────────────────────────────────────
    best_entry = max(fold_results, key=lambda r: r[1])
    best_fold_idx, best_f1, best_norm, best_ckpt, _ = best_entry
    logger.log(f"\n  Using Fold {best_fold_idx+1} checkpoint "
               f"(F1={best_f1:.6f}) for test inference.")

    # ── Test inference ────────────────────────────────────────────────────────
    logger.log("\n--- Test Inference ---")
    submission = predict_test(best_ckpt, best_norm, CFG, device)
    submission.to_csv(CFG["submission_path"], index=False)

    logger.log(f"  Submission saved to : {CFG['submission_path']}")
    logger.log(f"  Total test samples  : {len(submission)}")
    pred_dist = submission["Label"].value_counts().sort_index()
    logger.log("  Predicted label distribution on test set:")
    for lbl, cnt in pred_dist.items():
        logger.log(f"    Label {lbl}: {cnt:>5d}  ({cnt / len(submission) * 100:.1f}%)")

    logger.log("\n" + "=" * 70)
    logger.log("  Pipeline complete. Submission file ready.")
    logger.log("=" * 70)

    logger.close()


if __name__ == "__main__":
    main()
