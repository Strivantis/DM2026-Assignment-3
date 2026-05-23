"""
run_stacking.py
---------------
Post-Phase-A stacking pipeline for v13.

Picks up from already-saved Phase A checkpoints and executes:
  Phase A'  – Re-generate OOF probs + test probs from saved checkpoints (no re-training)
  Phase B   – Tabular feature extraction (Jerk + Rolling Variation CV, FFT pruned)
  Phase C   – LightGBM meta-learner with native multiclass + Optuna tuning (25 trials/fold)
  Output    – submission_v13_stacking.csv

v13 changes vs v12:
  ✗ Custom Focal Loss objective & _LGBM_FOCAL_GAMMA – fully removed
  ✗ build_meta_lgbm_model() – replaced by Optuna-driven model construction
  ✓ Native objective='multiclass' (stable cross-entropy)
  ✓ sklearn compute_class_weight('balanced') + dynamic M_L2 ∈ [1.0, 5.0]
  ✓ Optuna TPE Bayesian search: 25 trials per fold

Prerequisites (all 5 checkpoints must exist):
  ./checkpoints/v12_fold_1_best.pth
  ./checkpoints/v12_fold_2_best.pth
  ./checkpoints/v12_fold_3_best.pth
  ./checkpoints/v12_fold_4_best.pth
  ./checkpoints/v12_fold_5_best.pth

Usage:
    conda activate PyTorch
    python run_stacking.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ── Import shared components from train.py ────────────────────────────────────
# (train.py has no side effects at import time – argparse only runs under __main__)
from train import (
    seed_everything,
    DualLogger,
    format_confusion_matrix,
    NUM_CLASSES,
    N_TAB_FEATURES,
    N_META_FEATURES,
    CLASS_NAMES,
    extract_tabular_features,
    train_meta_lgbm,
)
from dataset import (
    load_all_samples,
    load_test_samples,
    get_fold_splits,
    normalize,
    compute_normalization_params,
    N_FOLDS,
)
from model import InceptionTime1D

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
SCFG = {
    "train_root"      : "./train/train",
    "test_root"       : "./test/test",
    "checkpoint_dir"  : "./checkpoints",
    "submission_path" : "./submission_v13_stacking.csv",
    "log_path"        : "./stacking_log.txt",
    # Model architecture  (must match Phase A training)
    "in_channels"     : 8,
    "nb_filters"      : 64,
    "bottleneck"      : 32,
    "depth"           : 6,
    "se_reduction"    : 16,
    "num_classes"     : NUM_CLASSES,
    # DataLoader
    "batch_size"      : 256,   # larger batch = faster inference
    "num_workers"     : 4,
    "pin_memory"      : True,
    "seed"            : 42,
    # Optuna
    "optuna_n_trials" : 25,    # search budget per fold
}


# ══════════════════════════════════════════════════════════════════════════════
# Phase A' : Re-generate OOF + test probabilities from saved checkpoints
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _infer_probs(model: torch.nn.Module,
                 signals_CL: np.ndarray,
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 device: torch.device,
                 tta: bool = False) -> np.ndarray:
    """
    Run inference on a (N, 8, 300) signal array.

    Parameters
    ----------
    tta : bool  If True, averages predictions over ×0.95 / ×1.00 / ×1.05 scales.

    Returns
    -------
    probs : np.ndarray, shape (N, NUM_CLASSES)
    """
    N = signals_CL.shape[0]
    tensor  = torch.from_numpy(signals_CL.astype(np.float32))
    dummy   = torch.zeros(N, dtype=torch.long)
    ds      = TensorDataset(tensor, dummy)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

    model.eval()
    all_probs = []
    for x_batch, _ in loader:
        x = x_batch.to(device, non_blocking=True)
        if tta:
            p_orig = F.softmax(model(x),        dim=1).cpu().numpy()
            p_up   = F.softmax(model(x * 1.05), dim=1).cpu().numpy()
            p_down = F.softmax(model(x * 0.95), dim=1).cpu().numpy()
            all_probs.append((p_orig + p_up + p_down) / 3.0)
        else:
            all_probs.append(F.softmax(model(x), dim=1).cpu().numpy())

    return np.concatenate(all_probs, axis=0)   # (N, NUM_CLASSES)


def collect_oof_and_test_probs(all_signals: np.ndarray,
                                all_labels:  np.ndarray,
                                all_groups:  np.ndarray,
                                global_mean: np.ndarray,
                                global_std:  np.ndarray,
                                normed_test_CL: np.ndarray,
                                device:      torch.device,
                                logger:      DualLogger) -> tuple:
    """
    Load each fold's best checkpoint, run inference on its validation fold (OOF)
    and on the test set (with TTA), then aggregate.

    Returns
    -------
    oof_cnn_probs  : np.ndarray (N_train, NUM_CLASSES)  – OOF probabilities
    oof_idx_covered: np.ndarray (N_train,)  bool mask – slots filled
    cnn_test_probs : np.ndarray (N_test, NUM_CLASSES)   – averaged TTA test probs
    """
    N_train = len(all_labels)
    N_test  = normed_test_CL.shape[0]

    oof_probs_full  = np.zeros((N_train, NUM_CLASSES), dtype=np.float32)
    oof_idx_covered = np.zeros(N_train, dtype=bool)
    test_prob_accum = np.zeros((N_test,  NUM_CLASSES), dtype=np.float64)
    n_folds_loaded  = 0

    for fold_idx in range(N_FOLDS):
        fold_num  = fold_idx + 1
        ckpt_path = os.path.join(SCFG["checkpoint_dir"],
                                  f"v12_fold_{fold_num}_best.pth")

        if not os.path.exists(ckpt_path):
            logger.log(f"  [WARNING] Checkpoint not found, skipping: {ckpt_path}")
            continue

        logger.log(f"  [Fold {fold_num}] Loading checkpoint: {ckpt_path}")

        # ── Build model and load weights ───────────────────────────────────────
        model = InceptionTime1D(
            in_channels  = SCFG["in_channels"],
            num_classes  = SCFG["num_classes"],
            nb_filters   = SCFG["nb_filters"],
            bottleneck   = SCFG["bottleneck"],
            depth        = SCFG["depth"],
            se_reduction = SCFG["se_reduction"],
        ).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        # ── Get validation split indices (must match training splits) ──────────
        _, val_idx = get_fold_splits(all_signals, all_labels, all_groups, fold_idx)

        # ── Normalise validation signals (global params) ───────────────────────
        val_signals_norm    = normalize(all_signals[val_idx], global_mean, global_std)
        val_signals_norm_CL = val_signals_norm.transpose(0, 2, 1).astype(np.float32)

        # ── OOF inference (no TTA needed here – deterministic) ────────────────
        t0 = time.time()
        oof_probs_fold = _infer_probs(
            model, val_signals_norm_CL,
            batch_size=SCFG["batch_size"], num_workers=SCFG["num_workers"],
            pin_memory=SCFG["pin_memory"], device=device, tta=False)
        oof_f1 = f1_score(all_labels[val_idx], oof_probs_fold.argmax(axis=1),
                           average="macro", zero_division=0)
        logger.log(f"  [Fold {fold_num}] OOF Val Macro F1 (from ckpt): {oof_f1:.4f}  "
                   f"({time.time() - t0:.1f}s)")

        oof_probs_full[val_idx]  = oof_probs_fold
        oof_idx_covered[val_idx] = True

        # ── Test inference with TTA ────────────────────────────────────────────
        t0 = time.time()
        test_probs_fold = _infer_probs(
            model, normed_test_CL,
            batch_size=SCFG["batch_size"], num_workers=SCFG["num_workers"],
            pin_memory=SCFG["pin_memory"], device=device, tta=True)
        test_prob_accum += test_probs_fold.astype(np.float64)
        n_folds_loaded  += 1
        logger.log(f"  [Fold {fold_num}] TTA test probs accumulated  ({time.time() - t0:.1f}s)")

    if n_folds_loaded == 0:
        raise RuntimeError("No checkpoints loaded. Ensure v12_fold_*_best.pth files exist.")

    cnn_test_probs = (test_prob_accum / n_folds_loaded).astype(np.float32)
    logger.log(f"\n  OOF slots filled  : {oof_idx_covered.sum()} / {N_train}")
    logger.log(f"  Folds loaded      : {n_folds_loaded}")
    return oof_probs_full, oof_idx_covered, cnn_test_probs


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    seed_everything(SCFG["seed"])
    logger = DualLogger(SCFG["log_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log("=" * 70)
    logger.log("  run_stacking.py – v13 Phase B+C from saved Phase A checkpoints")
    logger.log("  (Native Multiclass | Optuna M_L2 Tuning | No SMOTE | No Custom Loss)")
    logger.log(f"  Device     : {device}")
    if device.type == "cuda":
        logger.log(f"  GPU        : {torch.cuda.get_device_name(0)}")
    logger.log(f"  Checkpoints: {SCFG['checkpoint_dir']}/v12_fold_{{1..5}}_best.pth")
    logger.log(f"  Submission : {SCFG['submission_path']}")
    logger.log(f"  Optuna     : {SCFG['optuna_n_trials']} trials per meta-fold")
    logger.log("=" * 70)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 0: Data loading & global normalisation
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n══ STEP 0: Data Loading & Global Normalisation ══")
    t0 = time.time()

    all_signals, all_labels, all_groups, _ = load_all_samples(SCFG["train_root"])
    global_mean, global_std = compute_normalization_params(all_signals)

    raw_test_signals, test_file_ids = load_test_samples(SCFG["test_root"])
    normed_test_signals = normalize(raw_test_signals, global_mean, global_std)
    normed_test_CL      = normed_test_signals.transpose(0, 2, 1).astype(np.float32)

    n_train, n_test = len(all_labels), len(test_file_ids)
    logger.log(f"  Train: {n_train}  Test: {n_test}  ({time.time() - t0:.1f}s)")
    logger.log(f"  Label distribution:")
    for c in range(NUM_CLASSES):
        cnt = int((all_labels == c).sum())
        logger.log(f"    {CLASS_NAMES[c]}: {cnt:>5d}  ({cnt / n_train * 100:.1f}%)")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A' : OOF + test probs from saved checkpoints
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n══ PHASE A' : Re-generate OOF + Test Probs from Checkpoints ══")
    t_a = time.time()

    oof_cnn_probs, oof_idx_covered, cnn_test_probs = collect_oof_and_test_probs(
        all_signals=all_signals,
        all_labels=all_labels,
        all_groups=all_groups,
        global_mean=global_mean,
        global_std=global_std,
        normed_test_CL=normed_test_CL,
        device=device,
        logger=logger,
    )

    # Aggregate OOF stats
    oof_preds_all  = oof_cnn_probs[oof_idx_covered].argmax(axis=1)
    oof_labels_all = all_labels[oof_idx_covered]
    oof_f1_all     = f1_score(oof_labels_all, oof_preds_all,
                               average="macro", zero_division=0)
    logger.log(f"\n  Combined OOF Macro F1 (all folds): {oof_f1_all:.4f}")
    cm_text = format_confusion_matrix(
        oof_labels_all.tolist(), oof_preds_all.tolist(), CLASS_NAMES)
    logger.log("--- OOF Confusion Matrix (CNN only) ---")
    logger.log(cm_text)
    logger.log(f"  Phase A' elapsed: {(time.time() - t_a) / 60:.1f} min")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE B : Tabular feature extraction
    # ════════════════════════════════════════════════════════════════════════
    logger.log(f"\n══ PHASE B: Tabular Feature Extraction ({N_TAB_FEATURES} features) ══")
    t_b = time.time()

    all_signals_norm    = normalize(all_signals, global_mean, global_std)
    all_signals_norm_CL = all_signals_norm.transpose(0, 2, 1).astype(np.float32)

    logger.log(f"  Extracting train features ({all_signals_norm_CL.shape}) ...")
    t_tr = time.time()
    tab_feats_train = extract_tabular_features(all_signals_norm_CL)
    logger.log(f"  Train: {tab_feats_train.shape}  ({time.time() - t_tr:.1f}s)")

    normed_test_CL_feat = normed_test_signals.transpose(0, 2, 1).astype(np.float32)
    logger.log(f"  Extracting test  features ({normed_test_CL_feat.shape}) ...")
    t_te = time.time()
    tab_feats_test = extract_tabular_features(normed_test_CL_feat)
    logger.log(f"  Test : {tab_feats_test.shape}  ({time.time() - t_te:.1f}s)")

    logger.log(f"  Phase B elapsed: {time.time() - t_b:.1f}s")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE C : Meta-feature assembly + LightGBM stacking (v13 Optuna)
    # ════════════════════════════════════════════════════════════════════════
    logger.log(f"\n══ PHASE C: Meta-Feature Assembly + LightGBM v13 (Optuna) ══")

    train_mask   = oof_idx_covered
    n_meta_train = int(train_mask.sum())

    if n_meta_train == 0:
        raise RuntimeError("No OOF-covered samples. Are all 5 checkpoints present?")

    # (N, 6) OOF CNN probs  +  (N, 136) tabular  →  (N, 142)
    meta_X_train = np.concatenate([
        oof_cnn_probs[train_mask],
        tab_feats_train[train_mask],
    ], axis=1).astype(np.float32)

    meta_y_train      = all_labels[train_mask]
    meta_groups_train = all_groups[train_mask]

    # (N_test, 6) + (N_test, 136) → (N_test, 142)
    meta_X_test = np.concatenate([
        cnn_test_probs,
        tab_feats_test,
    ], axis=1).astype(np.float32)

    logger.log(f"  Meta-train: {meta_X_train.shape}   Meta-test: {meta_X_test.shape}")
    logger.log(f"  Label distribution (meta-train):")
    for c in range(NUM_CLASSES):
        cnt = int((meta_y_train == c).sum())
        logger.log(f"    {CLASS_NAMES[c]}: {cnt:>5d}  ({cnt / n_meta_train * 100:.1f}%)")

    t_c = time.time()
    test_preds = train_meta_lgbm(
        meta_X_train              = meta_X_train,
        meta_y_train              = meta_y_train,
        meta_X_test               = meta_X_test,
        all_signals_train_norm_CL = all_signals_norm_CL[train_mask],
        all_labels                = meta_y_train,
        folds_to_run              = list(range(N_FOLDS)),
        all_groups                = meta_groups_train,
        cfg                       = SCFG,
        logger                    = logger,
    )
    logger.log(f"\n  Phase C elapsed: {(time.time() - t_c) / 60:.1f} min")

    # ════════════════════════════════════════════════════════════════════════
    # Submission assembly
    # ════════════════════════════════════════════════════════════════════════
    logger.log("\n══ SUBMISSION ASSEMBLY ══")
    submission = (
        pd.DataFrame({"Id": test_file_ids, "Label": test_preds})
        .sort_values("Id")
        .reset_index(drop=True)
    )
    submission.to_csv(SCFG["submission_path"], index=False)
    logger.log(f"  Saved → {SCFG['submission_path']}")

    pred_dist = submission["Label"].value_counts().sort_index()
    for lbl, cnt in pred_dist.items():
        lbl_name = CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else str(lbl)
        logger.log(f"    {lbl_name} (Label {lbl}): {cnt:>5d}  "
                   f"({cnt / n_test * 100:.1f}%)")

    total = time.time() - t0
    logger.log(f"\n  Total wall-clock: {total / 60:.1f} min")
    logger.log("=" * 70)
    logger.close()


if __name__ == "__main__":
    main()
