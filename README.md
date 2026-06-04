# HAR SE-InceptionTime OOF Stacking Pipeline — v13

Human Activity Recognition via a 3-phase pipeline:  
**Phase A** SE-InceptionTime base model (5-fold OOF) → **Phase B** handcrafted tabular features → **Phase C** LightGBM meta-learner (Optuna-tuned).

Public leaderboard Macro F1: **0.8078**

---

## Table of Contents

1. [Environment Setup & Requirements](#1-environment-setup--requirements)
2. [Data Directory Structure](#2-data-directory-structure)
3. [Script Reference & Evaluation Documentation](#3-script-reference--evaluation-documentation)

---

## 1. Environment Setup & Requirements

### Core Dependencies

The following packages are required. All are used directly by `train.py`, `dataset.py`, or `model.py`:

| Package | Used for |
|---|---|
| `torch` / `torchvision` | SE-InceptionTime model, DataLoader, Focal Loss, Mixup |
| `numpy` | Array operations, feature extraction, OOF accumulation |
| `pandas` | CSV loading, submission assembly |
| `scipy` | `kurtosis`, `skew` in tabular feature extraction |
| `scikit-learn` | `StratifiedGroupKFold`, `f1_score`, `classification_report`, `compute_class_weight` |
| `lightgbm` | Phase C meta-learner |
| `optuna` | Bayesian hyperparameter search (25 trials/fold, Phase C) |

Additional packages required by `integrated_eda.py`:

| Package | Used for |
|---|---|
| `matplotlib` | All plot rendering (Agg backend) |
| `seaborn` | KDE plots, bar plots, scatter plots |
| `statsmodels` | ACF computation (`statsmodels.tsa.stattools.acf`) |
| `tqdm` | Progress bars during data loading |

### Verify & Install

Check installed packages:

```bash
conda activate PyTorch
pip list | grep -E "torch|numpy|pandas|scipy|scikit|lightgbm|optuna|matplotlib|seaborn|statsmodels|tqdm"
```

Install any missing package:

```bash
pip install lightgbm optuna statsmodels tqdm
```

---

## 2. Data Directory Structure

The pipeline expects data at the following paths relative to the project root (controlled by `CFG["train_root"]` and `CFG["test_root"]` in `train.py`):

```
Assignment3/
├── train/
│   └── train/                        ← CFG["train_root"] = "./train/train"
│       ├── User_001/                 ┐
│       │   ├── 00001.csv             │  60 subjects (User_001 – User_060)
│       │   ├── 00002.csv             │  ~196 CSV files per subject
│       │   └── ...                   │  11,020 total training samples
│       ├── User_002/                 │
│       │   └── ...                   │
│       └── User_060/                 ┘
│           └── ...
│
├── test/
│   └── test/                         ← CFG["test_root"] = "./test/test"
│       ├── User_061/                 ┐
│       │   ├── 11021.csv             │  40 subjects (User_061 – User_100)
│       │   └── ...                   │  6,849 total test samples
│       └── User_100/                 ┘
│           └── ...
│
├── checkpoints/                      ← v13 fold checkpoints written here
│   ├── v13_fold_1_best.pth
│   └── ...
├── plots/                            ← EDA and ablation charts written here
├── reports/                          ← Quantitative EDA reports written here
│
├── train.py
├── model.py
├── dataset.py
├── integrated_eda.py
├── experiment_preprocessing.py
├── ablation_phase_c.py
└── sample_submission.csv
```

### Notes

- `dataset.py`'s `load_all_samples()` scans `train/train/**/*.csv` recursively — the nested `train/train/` double directory is required.
- The test set has **no labels**; EDA is conducted on training data only.
- `checkpoints/` is created automatically by `train.py` on first run.
- `plots/` and `reports/` are created automatically by `integrated_eda.py`.

---

## 3. Script Reference & Evaluation Documentation

### `train.py` — Main Pipeline

```bash
python train.py
```

Runs all three phases end-to-end. Writes:
- `checkpoints/v13_fold_{n}_best.pth` (Phase A)
- `submission_v13_stacking.csv` (final predictions)
- `training_log.txt` (full per-epoch log)

Optional CLI arguments: `--fold-1-only` (quick single-fold run), `--epochs`, `--batch-size`, `--lr`, `--patience`, `--optuna-trials`.

---

### `integrated_eda.py` — Exploratory Data Analysis

```bash
python integrated_eda.py
```

Runs a 7-module EDA pipeline on the training set:

1. **Label distribution** — sequence-level class counts and imbalance ratios
2. **Deep EDA** — t-SNE manifold projection (6 classes), Random Forest feature importance, user heterogeneity
3. **Quantitative report** — Gini importances (CSV), feature collinearity, cross-user covariate shift (CV)
4. **Feature space analysis** — KDE densities and PCA for L1 vs. L2
5. **Spatial geometry** — covariance eigenvalue ratio distributions for L1 vs. L2
6. **Global ACF** — autocorrelation profiles (P10–P90) for L1 vs. L2
7. **Statistics summary** — CSV exports of global and per-class feature statistics

Outputs saved to `plots/` (5 PNG figures) and `reports/` (CSV + Markdown).

---

### `experiment_preprocessing.py` — Phase A Preprocessing Ablation

```bash
python experiment_preprocessing.py
```

Runs a controlled leave-one-out experiment on **Fold 1** to quantify the isolated impact of each Phase A preprocessing component on SE-InceptionTime Macro F1:

| Variation | Removed component |
|---|---|
| Variation 1 (Baseline) | None — full v13 stack |
| Variation 2 | Batch-level Mixup (α = 0.2) |
| Variation 3 | All 4× augmentations (aug_prob = 0.0) |
| Variation 4 | Global Z-score normalization |

Output is a console/log summary (`experiment_preprocessing.md`) containing the best Val Macro F1 and ΔF1 vs. baseline for each variation. These figures form the quantitative basis for **Section 2 (Preprocessing Techniques)** of `DM_asg3_513559009.md`.

Wall-clock: ~52 min on RTX 4070 Laptop GPU.

---

### `ablation_phase_c.py` — Phase C Feature Group Ablation

```bash
python ablation_phase_c.py              # full 5-fold ablation (~29 min)
python ablation_phase_c.py --plot-only  # regenerate charts from saved results
```

Runs a 5-fold CV ablation study to measure the individual and joint contribution of the `jerk` and `rolling_var_cv` feature groups to the LightGBM meta-learner:

| Config | Feature set | Dims |
|---|---|---|
| Config 1 | Base (time-domain + 1st-derivative) | 118 / 142 |
| Config 2 | Base + Rolling Variance CV | 126 / 142 |
| Config 3 | Base + Jerk | 134 / 142 |
| Config 4 | Full v13 (all features) | 142 / 142 |

Each config runs Optuna (5 trials/fold) over M_L2, max_depth, colsample_bytree, reg_lambda. Output is a console summary (`ablation_phase_c.md`) with per-fold Macro F1, aggregate OOF F1, and ΔF1 vs. Config 4. The `--plot-only` flag regenerates `plots/06_ablation_macro_f1_comparison.png`, `plots/07_ablation_per_fold_f1.png`, and `plots/08_ablation_marginal_contribution.png` from pre-computed results without re-running the experiment.

These results form the quantitative basis for **Section 4 (Ablation Study)** of `DM_asg3_513559009.md`.

Wall-clock: ~29 min on CPU (no GPU required).
