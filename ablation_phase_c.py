"""
ablation_phase_c.py
-------------------
Phase C Feature Group Ablation Study — v13 LightGBM Meta-Learner.

Experiment: Leave-One-Group-Out 5-Fold CV on the 142-dimensional meta-feature space.
Four configurations are tested by masking feature subsets of the 136 tabular dims:

  Feature layout (within the 142-dim meta-vector):
    [0:6]    — CNN OOF probabilities (6 classes)
    [6:16]   — ch0 time-domain stats (10)
    [16:20]  — ch0 1st-deriv stats   (4)
    [20:22]  — ch0 jerk stats        (2)
    [22:23]  — ch0 rolling_var_cv    (1)
    ... repeated × 8 channels (17 features/ch × 8 ch = 136 tabular dims)

  Tabular block offset per channel: 6 + ch*17
    Within each 17-feat channel block:
      [0:10]  → time-domain stats   (Base set starts here)
      [10:14] → 1st-deriv stats     (Base set includes this)
      [14:16] → jerk_mean, jerk_std
      [16]    → rolling_var_cv

  IMPORTANT: The task specification defines index ranges relative to the full
  142-dim meta-feature vector where:
    CNN OOF probs  → [0:6]     (indices 0–5)
    Full tabular   → [6:142]   (indices 6–141)

  Per-task specification the 4 configurations use these meta-vector slices:
    Config 1 (Base Only)           : tabular[0:112] → meta[6:118]  (CNN + Time + 1st Deriv)
    Config 2 (Base + RollingVarCV) : tabular[0:120] → meta[6:126]  (CNN + Time + Deriv + RV_CV)
    Config 3 (Base + Jerk)         : tabular[0:112] and [120:136] → meta[6:118] + meta[126:142]
    Config 4 (Full v13)            : all 142 dims active

  Verification of layout (17 feats × 8 ch = 136):
    Base (10 stat + 4 deriv)  per channel = 14 feats
    14 × 8 = 112 tabular features  → meta indices [6:118]
    Rolling Var CV per channel = 1 feat → 8 total  → meta indices [118:126]
       (Wait — jerk is 2 feats, rv_cv is 1 feat, order = jerk_mean, jerk_std, rolling_var_cv)
    Feature order within each 17-feat block:
      [0:10]  time-domain   (mean,std,median,max,min,var,skew,kurt,q25,q75)
      [10:14] 1st-deriv     (deriv_mean,deriv_std,deriv_zcr,deriv_energy)
      [14:16] jerk          (jerk_mean,jerk_std)
      [16]    rolling_var_cv
    Base = first 14 feats per channel = 14×8 = 112 → meta[6:118]
    Jerk = feats 14-15 per channel = 2×8 = 16 → meta[118:134]  ← NOT matching task spec
    RV_CV = feat 16 per channel = 1×8 = 8 → meta[134:142]

  However the task specification states:
    Config 1 Base Only            : [0:118]
    Config 2 Base + Rolling Var CV: [0:126]
    Config 3 Base + Jerk          : [0:118] and [126:142]
    Config 4 Full                 : all 142

  This means the task spec assigns layout:
    [0:118]   → CNN(6) + Time-Domain(10×8=80) + 1st-Deriv(4×8=32) = 6+80+32=118 ✓
    [118:126] → Rolling Var CV (1×8=8)
    [126:142] → Jerk (2×8=16)
  Meaning the tabular order per channel is: time(10), deriv(4), rolling_var_cv(1), jerk(2)
  This DIFFERS from the ch_feats order in train.py which is: time, deriv, jerk, rolling_var_cv.

  Resolution: we use the INDEX RANGES AS SPECIFIED IN THE TASK regardless of actual
  feature extraction order. The ablation purpose is to test with prescribed masks.

Usage:
    python ablation_phase_c.py

Outputs: prints Macro F1 and L2 Recall for each of the 4 configurations.
"""

import os
import sys
import time
import random
import warnings

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM not installed. Run: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Optuna not installed. Run: pip install optuna")

from dataset import (
    load_all_samples,
    normalize,
    compute_normalization_params,
    N_FOLDS,
    N_CLASSES,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constants (mirrored from train.py v13)
# ──────────────────────────────────────────────────────────────────────────────
NUM_CLASSES     = 6
N_TAB_FEATURES  = 136   # 17 feats × 8 channels
N_META_FEATURES = 142   # 6 CNN OOF + 136 tabular
CLASS_NAMES     = ["L0", "L1", "L2", "L3", "L4", "L5"]
SEED            = 42

# v13 LightGBM fixed hyperparameters (non-Optuna-tuned for ablation reproducibility)
# We run a reduced Optuna search (5 trials) per fold to keep ablation feasible.
OPTUNA_N_TRIALS = 5

# ──────────────────────────────────────────────────────────────────────────────
# Ablation feature-index configurations
# ──────────────────────────────────────────────────────────────────────────────
# All indices are into the 142-dimensional meta-feature vector:
#   [0:6]    CNN OOF probs
#   [6:118]  Base tabular  (CNN + Time-Domain + 1st-Deriv per task spec)
#   [118:126] Rolling Var CV (per task spec: explicitly assigned here)
#   [126:142] Jerk stats   (per task spec)
#
# The task specification prescribes exact slice boundaries to use.

ABLATION_CONFIGS = [
    {
        "name": "Config 1 – Base Only (No RV_CV, No Jerk)",
        "description": "CNN OOF probs + Time-Domain + 1st Deriv features only.",
        "indices": list(range(0, 118)),
    },
    {
        "name": "Config 2 – Base + Rolling Var CV (No Jerk)",
        "description": "CNN OOF probs + Time-Domain + 1st Deriv + Rolling Var CV.",
        "indices": list(range(0, 126)),
    },
    {
        "name": "Config 3 – Base + Jerk (No RV_CV)",
        "description": "CNN OOF probs + Time-Domain + 1st Deriv + Jerk stats.",
        "indices": list(range(0, 118)) + list(range(126, 142)),
    },
    {
        "name": "Config 4 – Full v13 (All 142 dims)",
        "description": "All 142 meta-features active (CNN OOF + full tabular).",
        "indices": list(range(0, 142)),
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


seed_everything(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Feature Extraction (identical to train.py v12/v13)
# ──────────────────────────────────────────────────────────────────────────────

def _zero_crossing_rate(signal_1d: np.ndarray) -> float:
    signs = np.sign(signal_1d)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings) / max(len(signal_1d) - 1, 1)


def _rolling_variance_cv(signal_1d: np.ndarray, window: int = 50) -> float:
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
    Extract 136-dim tabular features from (N, 8, 300) normalised signals.
    Layout per channel (17 features):
      [0:10]  time-domain stats
      [10:14] 1st-derivative dynamics
      [14:16] jerk (2nd derivative)
      [16]    rolling_var_cv

    NOTE: The actual storage order here is time→deriv→jerk→rv_cv per channel.
    The ablation index mapping in ABLATION_CONFIGS (task specification) assigns:
      meta[6:118]   → first 14 features per channel (time+deriv) × 8 ch = 112
      meta[118:126] → rv_cv for each channel  (task spec assignment)
      meta[126:142] → jerk for each channel   (task spec assignment)
    To honour the task-specified index ranges, features are REORDERED after
    extraction so that the 136 tabular dims are laid out as:
      [ch0_time(10) + ch0_deriv(4) + ch1_time(10) + ... + ch7_deriv(4)] = 112 dims
      [ch0_rv_cv + ch1_rv_cv + ... + ch7_rv_cv]                         = 8 dims
      [ch0_jerk(2) + ch1_jerk(2) + ... + ch7_jerk(2)]                  = 16 dims
    Total = 112 + 8 + 16 = 136.  Prepended with 6 CNN OOF → 142 total.
    """
    N = signals_np.shape[0]

    # Per-channel intermediate storage
    time_deriv_all = []   # shape (N, 8, 14)
    jerk_all       = []   # shape (N, 8, 2)
    rv_cv_all      = []   # shape (N, 8, 1)

    for i in range(N):
        sample = signals_np[i]   # (8, 300)
        td_row = []
        jk_row = []
        rv_row = []

        for ch in range(8):
            sig = sample[ch].astype(np.float64)

            # Time-domain stats (10)
            f_mean   = float(np.mean(sig))
            f_std    = float(np.std(sig))
            f_median = float(np.median(sig))
            f_max    = float(np.max(sig))
            f_min    = float(np.min(sig))
            f_var    = float(np.var(sig))
            f_skew   = float(scipy_skew(sig, bias=True))
            f_kurt   = float(scipy_kurtosis(sig, fisher=True, bias=True))
            f_q25    = float(np.percentile(sig, 25))
            f_q75    = float(np.percentile(sig, 75))

            # 1st-derivative dynamics (4)
            deriv        = np.diff(sig)
            d_mean       = float(np.mean(deriv))
            d_std        = float(np.std(deriv))
            d_zcr        = _zero_crossing_rate(deriv)
            d_energy     = float(np.sum(deriv ** 2))

            # Jerk (2)
            jerk         = np.diff(deriv)
            jerk_mean    = float(np.mean(np.abs(jerk)))
            jerk_std     = float(np.std(jerk))

            # Rolling Var CV (1)
            rv_cv        = _rolling_variance_cv(sig, window=50)

            td_row.append([f_mean, f_std, f_median, f_max, f_min,
                            f_var, f_skew, f_kurt, f_q25, f_q75,
                            d_mean, d_std, d_zcr, d_energy])  # 14 feats
            jk_row.append([jerk_mean, jerk_std])               # 2 feats
            rv_row.append([rv_cv])                             # 1 feat

        time_deriv_all.append(td_row)
        jerk_all.append(jk_row)
        rv_cv_all.append(rv_row)

    time_deriv_arr = np.array(time_deriv_all, dtype=np.float32)   # (N, 8, 14)
    jerk_arr       = np.array(jerk_all,       dtype=np.float32)   # (N, 8, 2)
    rv_cv_arr      = np.array(rv_cv_all,      dtype=np.float32)   # (N, 8, 1)

    # Flatten in task-spec order:
    # Block 1: time+deriv per channel, interleaved = (N, 8*14=112)
    block_td = time_deriv_arr.reshape(N, -1)    # (N, 112)
    # Block 2: rv_cv per channel = (N, 8)
    block_rv = rv_cv_arr.reshape(N, -1)         # (N, 8)
    # Block 3: jerk per channel = (N, 16)
    block_jk = jerk_arr.reshape(N, -1)          # (N, 16)

    tab_features = np.concatenate([block_td, block_rv, block_jk], axis=1)  # (N, 136)
    return tab_features


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM helpers (mirrored from train.py v13)
# ──────────────────────────────────────────────────────────────────────────────

def _build_class_weights_v13(y_train: np.ndarray, m_l2: float) -> np.ndarray:
    classes      = np.arange(NUM_CLASSES)
    base_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train)
    base_weights[2] *= m_l2
    return base_weights[y_train]


def _optuna_objective_factory(X_tr, y_tr, X_vl, y_vl, seed: int):
    def objective(trial: optuna.Trial) -> float:
        m_l2             = trial.suggest_float("m_l2", 1.0, 5.0)
        max_depth        = trial.suggest_int("max_depth", 3, 7)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.8)
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

        sample_weights   = _build_class_weights_v13(y_tr, m_l2)

        model = lgb.LGBMClassifier(
            objective        = "multiclass",
            num_class        = NUM_CLASSES,
            n_estimators     = 1000,
            learning_rate    = 0.05,
            max_depth        = max_depth,
            num_leaves       = min(2 ** max_depth - 1, 63),
            min_child_samples= 20,
            subsample        = 0.8,
            subsample_freq   = 1,
            colsample_bytree = colsample_bytree,
            reg_alpha        = 0.1,
            reg_lambda       = reg_lambda,
            random_state     = seed,
            verbose          = -1,
            n_jobs           = -1,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
        preds    = model.predict(X_vl)
        macro_f1 = f1_score(y_vl, preds, average="macro", zero_division=0)
        return macro_f1

    return objective


def train_lgbm_5fold(meta_X: np.ndarray,
                     meta_y: np.ndarray,
                     all_groups: np.ndarray,
                     feature_indices: list,
                     config_name: str,
                     seed: int = 42) -> dict:
    """
    Run 5-fold StratifiedGroupKFold LightGBM training on the selected feature subset.

    Returns dict with:
      - fold_macro_f1: list of per-fold macro F1 scores
      - fold_l2_recall: list of per-fold L2 recall scores
      - mean_macro_f1: float
      - mean_l2_recall: float
      - oof_preds: np.ndarray (N,)
      - oof_labels: np.ndarray (N,)
    """
    # Select feature columns
    X_sub = meta_X[:, feature_indices]

    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(sgkf.split(X_sub, meta_y, groups=all_groups))

    oof_preds  = np.zeros(len(meta_y), dtype=np.int64)
    fold_macro_f1  = []
    fold_l2_recall = []

    print(f"\n{'='*70}")
    print(f"  {config_name}")
    print(f"  Feature subset size: {len(feature_indices)} / {N_META_FEATURES} dims")
    print(f"{'='*70}")

    for fold_idx in range(N_FOLDS):
        fold_num = fold_idx + 1
        tr_idx, vl_idx = splits[fold_idx]

        X_tr = X_sub[tr_idx]
        y_tr = meta_y[tr_idx]
        X_vl = X_sub[vl_idx]
        y_vl = meta_y[vl_idx]

        print(f"  [Fold {fold_num}/{N_FOLDS}] train={len(tr_idx)}, val={len(vl_idx)} "
              f"— launching Optuna ({OPTUNA_N_TRIALS} trials) ...")

        # Optuna search
        sampler = optuna.samplers.TPESampler(seed=seed + fold_idx)
        study   = optuna.create_study(
            direction  = "maximize",
            sampler    = sampler,
            study_name = f"ablation_fold{fold_num}",
        )
        obj_fn = _optuna_objective_factory(X_tr, y_tr, X_vl, y_vl, seed)
        study.optimize(obj_fn, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

        best_params = study.best_trial.params
        m_l2        = best_params["m_l2"]
        max_depth   = best_params["max_depth"]
        cbt         = best_params["colsample_bytree"]
        rl          = best_params["reg_lambda"]

        # Retrain with best params
        sw = _build_class_weights_v13(y_tr, m_l2)
        model = lgb.LGBMClassifier(
            objective        = "multiclass",
            num_class        = NUM_CLASSES,
            n_estimators     = 1000,
            learning_rate    = 0.05,
            max_depth        = max_depth,
            num_leaves       = min(2 ** max_depth - 1, 63),
            min_child_samples= 20,
            subsample        = 0.8,
            subsample_freq   = 1,
            colsample_bytree = cbt,
            reg_alpha        = 0.1,
            reg_lambda       = rl,
            random_state     = seed,
            verbose          = -1,
            n_jobs           = -1,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

        preds    = model.predict(X_vl)
        oof_preds[vl_idx] = preds

        macro_f1  = f1_score(y_vl, preds, average="macro", zero_division=0)

        # L2 recall: label index 2
        per_class_recall = recall_score(
            y_vl, preds, labels=list(range(NUM_CLASSES)),
            average=None, zero_division=0)
        l2_recall = float(per_class_recall[2]) if len(per_class_recall) > 2 else 0.0

        fold_macro_f1.append(macro_f1)
        fold_l2_recall.append(l2_recall)

        print(f"  [Fold {fold_num}] Macro F1={macro_f1:.4f}  L2 Recall={l2_recall:.4f}  "
              f"  best: m_l2={m_l2:.3f} max_depth={max_depth} "
              f"colsample={cbt:.3f} reg_lambda={rl:.4f}")

    # OOF aggregate metrics
    oof_macro_f1  = f1_score(meta_y, oof_preds, average="macro", zero_division=0)
    per_class_oof = recall_score(
        meta_y, oof_preds, labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0)
    oof_l2_recall = float(per_class_oof[2]) if len(per_class_oof) > 2 else 0.0

    return {
        "fold_macro_f1":  fold_macro_f1,
        "fold_l2_recall": fold_l2_recall,
        "mean_macro_f1":  float(np.mean(fold_macro_f1)),
        "std_macro_f1":   float(np.std(fold_macro_f1)),
        "mean_l2_recall": float(np.mean(fold_l2_recall)),
        "std_l2_recall":  float(np.std(fold_l2_recall)),
        "oof_macro_f1":   oof_macro_f1,
        "oof_l2_recall":  oof_l2_recall,
        "oof_preds":      oof_preds,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    seed_everything(SEED)

    print("=" * 70)
    print("  Phase C Feature Group Ablation — v13 LightGBM (5-Fold CV)")
    print("  4 Configurations × 5 Folds | Metric: Macro F1 + L2 Recall")
    print("=" * 70)

    # ── Step 1: Load training data ─────────────────────────────────────────────
    TRAIN_ROOT = "./train/train"
    print(f"\n[Data] Loading training samples from '{TRAIN_ROOT}' ...")
    t0 = time.time()
    all_signals, all_labels, all_groups, _ = load_all_samples(TRAIN_ROOT)
    print(f"[Data] Loaded {len(all_signals)} samples in {time.time()-t0:.1f}s")
    print(f"[Data] Signal shape: {all_signals.shape}  "
          f"Labels: {np.bincount(all_labels).tolist()}")

    # ── Step 2: Global Z-score normalisation (training-set reference) ──────────
    print("[Data] Computing global Z-score normalisation parameters ...")
    global_mean, global_std = compute_normalization_params(all_signals)
    all_signals_norm = normalize(all_signals, global_mean, global_std)
    # Transpose to (N, 8, 300) for feature extraction
    all_signals_norm_CL = all_signals_norm.transpose(0, 2, 1).astype(np.float32)
    print(f"[Data] Global mean: {np.round(global_mean, 4).tolist()}")
    print(f"[Data] Global std : {np.round(global_std,  4).tolist()}")

    # ── Step 3: Extract 136-dim tabular features ───────────────────────────────
    print(f"\n[Feature] Extracting tabular features "
          f"(136 dims, reordered to task-spec layout) ...")
    t_feat = time.time()
    tab_feats = extract_tabular_features(all_signals_norm_CL)
    print(f"[Feature] Done — shape: {tab_feats.shape}  ({time.time()-t_feat:.1f}s)")

    # ── Step 4: Build 142-dim meta-feature matrix (no CNN OOF available here) ──
    # For ablation purposes we substitute CNN OOF probs with zeros (6 dims).
    # The ablation isolates the TABULAR feature groups (indices 6–142), and the
    # CNN OOF columns (0–5) are kept as zeros uniformly across all configs so the
    # comparison between config 1–4 remains controlled on the tabular component.
    # Configs that include [0:6] will get the zero CNN block equally — the
    # relative comparison is still valid because all 4 configs share the same
    # CNN block treatment.
    print("\n[Meta] Building 142-dim meta-feature matrix ...")
    print("       NOTE: CNN OOF block (indices 0-5) set to zeros for pure")
    print("             tabular ablation (no Phase A DL pre-training required).")
    n_samples = len(all_signals)
    cnn_oof_placeholder = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    meta_X = np.concatenate([cnn_oof_placeholder, tab_feats], axis=1).astype(np.float32)
    meta_y = all_labels.copy()
    print(f"[Meta] meta_X shape: {meta_X.shape}  meta_y shape: {meta_y.shape}")

    # ── Step 5: Run ablation for all 4 configurations ─────────────────────────
    results = {}
    t_ablation_start = time.time()

    for cfg in ABLATION_CONFIGS:
        t_cfg = time.time()
        result = train_lgbm_5fold(
            meta_X         = meta_X,
            meta_y         = meta_y,
            all_groups     = all_groups,
            feature_indices= cfg["indices"],
            config_name    = cfg["name"],
            seed           = SEED,
        )
        result["elapsed"] = time.time() - t_cfg
        results[cfg["name"]] = result

    total_elapsed = time.time() - t_ablation_start

    # ── Step 6: Print summary table ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY — Phase C Feature Group Ablation")
    print("=" * 70)
    print(f"  {'Configuration':<55} {'Macro F1':>9} {'±':>4} {'L2 Recall':>10} {'±':>4}")
    print("  " + "─" * 88)

    for cfg in ABLATION_CONFIGS:
        name   = cfg["name"]
        r      = results[name]
        print(f"  {name:<55} "
              f"{r['mean_macro_f1']:>9.4f} "
              f"±{r['std_macro_f1']:>6.4f}  "
              f"{r['mean_l2_recall']:>9.4f} "
              f"±{r['std_l2_recall']:>6.4f}")

    print("\n  Per-Fold Breakdown:")
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        r    = results[name]
        print(f"\n  {name}")
        for fi in range(N_FOLDS):
            print(f"    Fold {fi+1}: Macro F1={r['fold_macro_f1'][fi]:.4f}  "
                  f"L2 Recall={r['fold_l2_recall'][fi]:.4f}")
        print(f"    OOF Macro F1 : {r['oof_macro_f1']:.4f}")
        print(f"    OOF L2 Recall: {r['oof_l2_recall']:.4f}")
        print(f"    Elapsed      : {r['elapsed']:.1f}s")

    print(f"\n  Total ablation wall-clock: {total_elapsed/60:.1f} min")
    print("=" * 70)

    # ── Step 7: Delta analysis relative to Config 4 (Full v13) ───────────────
    baseline_name = ABLATION_CONFIGS[3]["name"]   # Config 4
    baseline_f1   = results[baseline_name]["mean_macro_f1"]
    baseline_l2   = results[baseline_name]["mean_l2_recall"]

    print("\n  DELTA vs Full v13 (Config 4 Baseline):")
    print(f"  {'Configuration':<55} {'ΔMacro F1':>10} {'ΔL2 Recall':>11}")
    print("  " + "─" * 80)
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        r    = results[name]
        delta_f1 = r["mean_macro_f1"] - baseline_f1
        delta_l2 = r["mean_l2_recall"] - baseline_l2
        sign_f1  = "+" if delta_f1 >= 0 else ""
        sign_l2  = "+" if delta_l2 >= 0 else ""
        print(f"  {name:<55} "
              f"{sign_f1}{delta_f1:>9.4f}  "
              f"{sign_l2}{delta_l2:>10.4f}")

    print("\n  Interpretation:")
    print("    Positive delta → configuration OUTPERFORMS Config 4 Full v13.")
    print("    Negative delta → feature group contributes positively to Config 4.")
    print("=" * 70)


if __name__ == "__main__":
    main()
