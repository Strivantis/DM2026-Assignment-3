"""
inference_only.py
-----------------
Standalone inference script for HAR InceptionTime v9.

Skips Phase A + B training completely.  Instead:
  1. Loads existing Stage 1 and Stage 2 InceptionTime checkpoints from disk.
  2. Re-fits the LightGBM / XGBoost GBDT models for each fold using the
     saved training data (tabular feature extraction only – takes seconds).
  3. Runs the full TTFN + Dual-Engine hierarchical inference pipeline.
  4. Writes submission_v9_inception.csv.

Best-F1 scores (hardcoded from training_log.txt of the completed v9 run):
  Stage 1:  Fold1=0.753344  Fold2=0.805628  Fold3=0.756066
            Fold4=0.802235  Fold5=0.790199
  Stage 2:  Fold1=0.651235  Fold2=0.742923  Fold3=0.711546
            Fold4=0.609022  Fold5=0.645225

Usage
─────
    /home/sean/miniconda3/envs/PyTorch/bin/python inference_only.py
"""

import os
import sys
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

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
        raise ImportError("Install LightGBM or XGBoost first.")
else:
    XGB_AVAILABLE = False

from dataset import (
    build_fold_datasets, load_test_samples,
    N_FOLDS, N_CLASSES_S1,
)
from model import InceptionTime1D

warnings.filterwarnings("ignore", category=UserWarning)

# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(42)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "train_root"    : "./train/train",
    "test_root"     : "./test/test",
    "checkpoint_dir": "./checkpoints",
    "submission_path": "./submission_v9_inception.csv",
    "in_channels"   : 8,
    "nb_filters"    : 32,
    "bottleneck"    : 32,
    "depth"         : 6,
    "num_classes_s1": N_CLASSES_S1,   # 5
    "batch_size"    : 256,
    "num_workers"   : 4,
    "pin_memory"    : True,
    "s2_threshold"  : 0.5,
    "cnn_weight"    : 0.5,
    "gbdt_weight"   : 0.5,
}

# Best F1 scores from the completed training run – used for F1-weighted ensembling
S1_BEST_F1 = {1: 0.753344, 2: 0.805628, 3: 0.756066, 4: 0.802235, 5: 0.790199}
S2_BEST_F1 = {1: 0.651235, 2: 0.742923, 3: 0.711546, 4: 0.609022, 5: 0.645225}

# ── Tabular feature helpers ───────────────────────────────────────────────────
STD_MAG_CHANNEL = 7

def zero_crossing_rate(sig):
    signs = np.sign(sig)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    return float(np.sum(np.diff(signs) != 0)) / max(len(sig) - 1, 1)

def extract_tabular_features(signals_np):
    """signals_np: (N, 8, 300) → (N, 23)"""
    rows = []
    for i in range(signals_np.shape[0]):
        s = signals_np[i]          # (8, 300)
        m = s[STD_MAG_CHANNEL]     # (300,)
        row = np.array([
            float(np.percentile(m, 10)),
            float(np.percentile(m, 25)),
            float(np.percentile(m, 75)),
            float(np.percentile(m, 90)),
            float(scipy_kurtosis(m, fisher=True, bias=True)),
            float(scipy_skew(m, bias=True)),
            zero_crossing_rate(m),
            *s.mean(axis=1).tolist(),
            *s.std(axis=1).tolist(),
        ], dtype=np.float32)
        rows.append(row)
    return np.stack(rows, axis=0)

def build_gbdt_model():
    if LGBM_AVAILABLE:
        m = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1)
        return m, "LightGBM"
    else:
        m = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        return m, "XGBoost"


def get_ordered_signals(ds, batch_size, num_workers):
    """Iterate a HARDataset with shuffle=False → (signals_np (N,8,300), labels_np (N,))."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    sigs, lbls = [], []
    for x, y in loader:
        sigs.append(x.numpy())
        lbls.append(y.numpy())
    return np.concatenate(sigs, axis=0), np.concatenate(lbls, axis=0)


# ── Re-fit GBDT for every Stage 2 fold ───────────────────────────────────────
def rebuild_gbdt_models():
    """Returns dict {fold_num: fitted_gbdt_model}"""
    print("\n[GBDT] Re-fitting GBDT models from training data ...")
    gbdt_models = {}
    for fold_idx in range(N_FOLDS):
        fold_num = fold_idx + 1
        t0 = time.time()
        train_ds, val_ds, _, class_counts, _ = build_fold_datasets(
            CFG["train_root"], fold_idx=fold_idx, mode="stage2")

        if len(train_ds) == 0:
            print(f"  Fold {fold_num}: no L1/L2 samples – skipping GBDT.")
            gbdt_models[fold_num] = None
            continue

        train_sigs, train_lbls = get_ordered_signals(
            train_ds, CFG["batch_size"], CFG["num_workers"])

        X_train = extract_tabular_features(train_sigs)   # (N, 23)

        # Inverse-frequency sample weights
        sample_weight = None
        if class_counts[1] > 0:
            ratio = float(class_counts[0]) / float(class_counts[1])
            sample_weight = np.where(train_lbls == 1, ratio, 1.0).astype(np.float32)

        model, backend = build_gbdt_model()
        if sample_weight is not None:
            model.fit(X_train, train_lbls, sample_weight=sample_weight)
        else:
            model.fit(X_train, train_lbls)

        # Quick val check
        val_sigs, val_lbls = get_ordered_signals(
            val_ds, CFG["batch_size"], CFG["num_workers"])
        X_val = extract_tabular_features(val_sigs)
        val_f1 = f1_score(val_lbls, model.predict(X_val),
                          average="macro", zero_division=0)
        print(f"  Fold {fold_num}: {backend} val F1={val_f1:.4f}  "
              f"({len(train_lbls)} samples, {time.time()-t0:.1f}s)")

        gbdt_models[fold_num] = model

    return gbdt_models


# ── Main inference pipeline ───────────────────────────────────────────────────
@torch.no_grad()
def run_inference(gbdt_models: dict, device: torch.device):
    print("\n" + "=" * 70)
    print("  HIERARCHICAL INFERENCE  (Stage1 → Route → Stage2 Dual-Engine)")
    print("=" * 70)

    # ── TTFN: load raw test signals and self-normalise ────────────────────────
    print("\n  [TTFN] Loading test data and computing intrinsic Z-score ...")
    raw_300_8, file_ids = load_test_samples(CFG["test_root"])
    raw = raw_300_8.transpose(0, 2, 1).astype(np.float32)   # (N, 8, 300)
    n_test = raw.shape[0]

    flat       = raw.transpose(1, 0, 2).reshape(raw.shape[1], -1)   # (8, N*300)
    t_mean     = flat.mean(axis=1, keepdims=True)                    # (8, 1)
    t_std      = flat.std(axis=1,  keepdims=True)                    # (8, 1)
    t_std      = np.where(t_std < 1e-8, 1.0, t_std)
    normed     = ((raw - t_mean[np.newaxis]) / t_std[np.newaxis]).astype(np.float32)

    print(f"  Test N={n_test} | Per-ch mean: {t_mean.squeeze().round(4).tolist()}")
    print(f"  Per-ch std : {t_std.squeeze().round(4).tolist()}")

    normed_tensor = torch.from_numpy(normed)
    test_loader   = DataLoader(
        TensorDataset(normed_tensor, torch.zeros(n_test, dtype=torch.long)),
        batch_size=CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])

    # ── STEP 1: Stage 1 ensemble ─────────────────────────────────────────────
    print("\n  [Step 1] Stage 1 ensemble ...")
    s1_prob_sum = np.zeros((n_test, CFG["num_classes_s1"]), dtype=np.float64)
    s1_f1_total = 0.0

    for fold_num, best_f1 in S1_BEST_F1.items():
        ckpt = os.path.join(CFG["checkpoint_dir"], f"stage1_fold_{fold_num}_best.pth")
        if not os.path.isfile(ckpt):
            print(f"    S1 Fold {fold_num}: checkpoint not found – skipping.")
            continue
        print(f"    S1 Fold {fold_num}: {ckpt}  (F1={best_f1:.6f})")
        m = InceptionTime1D(CFG["in_channels"], CFG["num_classes_s1"],
                            CFG["nb_filters"], CFG["bottleneck"], CFG["depth"]).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()

        fold_p = []
        for x, _ in test_loader:
            x = x.to(device)
            p = (F.softmax(m(x), 1) + F.softmax(m(x*1.05), 1) + F.softmax(m(x*0.95), 1)) / 3.0
            fold_p.append(p.cpu().numpy())
        fold_p = np.concatenate(fold_p, axis=0)
        s1_prob_sum += best_f1 * fold_p
        s1_f1_total += best_f1

    s1_probs  = s1_prob_sum / s1_f1_total
    s1_preds  = s1_probs.argmax(axis=1)

    STAGE1_IDX_TO_ORIG = {0:0, 1:-1, 2:3, 3:4, 4:5}
    s2_mask   = (s1_preds == 1)
    n_routed  = int(s2_mask.sum())
    s1_dist   = {i: int((s1_preds==i).sum()) for i in range(CFG["num_classes_s1"])}
    print(f"\n  S1 distribution: {s1_dist}")
    print(f"  Resolved by S1: {n_test-n_routed}   Routed to S2: {n_routed}")

    final_preds = np.full(n_test, -1, dtype=np.int64)
    for idx, orig in STAGE1_IDX_TO_ORIG.items():
        if orig != -1:
            final_preds[s1_preds == idx] = orig

    # ── STEP 3: Stage 2 dual-engine ─────────────────────────────────────────
    if n_routed > 0:
        print("\n  [Step 3] Stage 2 Dual-Engine for routed samples ...")
        routed_np     = normed[s2_mask]                       # (n_routed, 8, 300)
        routed_tensor = torch.from_numpy(routed_np)
        routed_loader = DataLoader(TensorDataset(routed_tensor),
                                   batch_size=CFG["batch_size"]*2,
                                   shuffle=False, num_workers=0)

        # Engine A – CNN
        cnn_sum = np.zeros(n_routed, dtype=np.float64)
        cnn_total = 0.0
        for fold_num, best_f1 in S2_BEST_F1.items():
            ckpt = os.path.join(CFG["checkpoint_dir"], f"stage2_fold_{fold_num}_best.pth")
            if not os.path.isfile(ckpt) or best_f1 <= 0.0:
                continue
            print(f"    [EngA] S2 Fold {fold_num}: {ckpt}  (F1={best_f1:.6f})")
            m = InceptionTime1D(CFG["in_channels"], 1,
                                CFG["nb_filters"], CFG["bottleneck"], CFG["depth"]).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            fp = []
            for (x,) in routed_loader:
                x = x.to(device)
                p = (torch.sigmoid(m(x).squeeze(-1))
                     + torch.sigmoid(m(x*1.05).squeeze(-1))
                     + torch.sigmoid(m(x*0.95).squeeze(-1))) / 3.0
                fp.append(p.cpu().numpy())
            fp = np.concatenate(fp, axis=0)
            cnn_sum   += best_f1 * fp
            cnn_total += best_f1

        p_cnn = cnn_sum / cnn_total if cnn_total > 0 else np.full(n_routed, 0.5)

        # Engine B – GBDT
        X_tab  = extract_tabular_features(routed_np)          # (n_routed, 23)
        gbdt_sum  = np.zeros(n_routed, dtype=np.float64)
        gbdt_total = 0.0
        for fold_num, best_f1 in S2_BEST_F1.items():
            gm = gbdt_models.get(fold_num)
            if gm is None or best_f1 <= 0.0:
                continue
            print(f"    [EngB] GBDT Fold {fold_num}: predicting  (F1={best_f1:.6f})")
            p_g = gm.predict_proba(X_tab)[:, 1]
            gbdt_sum   += best_f1 * p_g
            gbdt_total += best_f1

        p_gbdt = gbdt_sum / gbdt_total if gbdt_total > 0 else np.full(n_routed, 0.5)

        # Weighted head
        p_final = CFG["cnn_weight"] * p_cnn + CFG["gbdt_weight"] * p_gbdt
        print(f"\n    CNN w={CFG['cnn_weight']}  GBDT w={CFG['gbdt_weight']}  "
              f"threshold={CFG['s2_threshold']}")

        s2_bin  = (p_final >= CFG["s2_threshold"]).astype(np.int64)
        s2_orig = np.where(s2_bin == 0, 1, 2)

        for i, idx in enumerate(np.where(s2_mask)[0]):
            final_preds[idx] = int(s2_orig[i])

        print(f"  S2 → L1={int((s2_orig==1).sum())}  L2={int((s2_orig==2).sum())}")
    else:
        print("  No samples routed to Stage 2.")

    # Sanity check
    unassigned = int((final_preds == -1).sum())
    if unassigned:
        print(f"  WARNING: {unassigned} unassigned → defaulting to 0")
        final_preds[final_preds == -1] = 0

    # ── Save submission ───────────────────────────────────────────────────────
    sub = (pd.DataFrame({"Id": file_ids, "Label": final_preds})
           .sort_values("Id").reset_index(drop=True))
    sub.to_csv(CFG["submission_path"], index=False)

    print(f"\n  ── Submission saved: {CFG['submission_path']} ──")
    dist = sub["Label"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"    Label {lbl}: {cnt:>5d}  ({cnt/n_test*100:.1f}%)")

    return sub


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    gbdt_models = rebuild_gbdt_models()
    t_gbdt = time.time() - t0
    print(f"\n  GBDT rebuild wall-clock: {t_gbdt:.1f}s")

    run_inference(gbdt_models, device)
    print(f"\n  Total wall-clock: {time.time()-t0:.1f}s")
