"""
integrated_eda.py
=================
Unified HAR (Human Activity Recognition) EDA pipeline.
Consolidates: eda.py, eda_deep.py, eda_deep_v2.py, eda_space.py, eda_geo.py, eda_acf.py

All plots are saved to ./plots/
All text/CSV reports are saved to ./reports/

Audit findings resolved:
  - eda.py had a dead `TEST_DIR` variable (test CSVs have no labels; EDA is train-only).
  - eda_deep.py and eda_deep_v2.py had duplicate load/feature logic — merged here.
  - eda_space.py / eda_geo.py / eda_acf.py silently only analyzed labels 1 & 2 — now
    configurable and explicitly documented.
  - plt.show() calls removed to prevent blocking in headless environments.
  - All savefig() calls standardized to PLOTS_DIR.
"""

import os
import glob
import random
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — prevents plt.show() blocking

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_DIR = "train/train"   # Path: train/train/User_NNN/*.csv  (has labels 0-5)
TEST_DIR  = "test/test"     # Path: test/test/User_NNN/*.csv    (NO labels — unlabeled test set)

PLOTS_DIR   = "plots"
REPORTS_DIR = "reports"

RANDOM_SEED        = 42
SAMPLES_PER_CLASS  = 800   # For t-SNE / RF (eda_deep)
MAX_BINARY_SAMPLES = 1500  # For binary class analyses (space, geo, acf)
ACF_NLAGS          = 50

LABEL_NAMES = {
    0: "L0",
    1: "L1 (Running)",
    2: "L2 (Stairs Up)",
    3: "L3",
    4: "L4",
    5: "L5",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs(*dirs: str) -> None:
    """Create output directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_all_csvs(base_path: str) -> list[str]:
    """Return sorted list of all CSV files under base_path/User_*/*.csv."""
    pattern = os.path.join(base_path, "User_*", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found matching: {pattern}\n"
            f"Check that TRAIN_DIR='{TRAIN_DIR}' is correct."
        )
    return files


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to PLOTS_DIR and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈 Saved plot → {path}")


# ============================================================
# MODULE 1: Basic EDA — label distribution + global statistics
# (from eda.py)
# ============================================================

def run_basic_eda(train_files: list[str]) -> pd.DataFrame:
    """
    Load all train CSVs, check NaN, validate label consistency,
    compute global statistics, and produce a label-distribution bar chart.

    Returns:
        full_train_df: Concatenated DataFrame of all train data.
    """
    print("\n" + "=" * 60)
    print("MODULE 1: Basic EDA — Label Distribution & Global Stats")
    print("=" * 60)

    df_list             = []
    file_label_mapping  = {}
    is_label_consistent = True
    has_nan             = False

    for file_path in tqdm(train_files, desc="Loading train CSVs"):
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"  ⚠️  Could not read {file_path}: {exc}")
            continue

        if df.isnull().values.any():
            has_nan = True

        unique_labels = df["label"].unique()
        file_id       = df["file_id"].iloc[0]

        if len(unique_labels) > 1:
            is_label_consistent = False

        file_label_mapping[file_id] = unique_labels[0]
        df_list.append(df)

    full_train_df = pd.concat(df_list, ignore_index=True)

    # ---- Console report ----
    print(f"\n  Total timesteps loaded : {len(full_train_df):,}")
    print(f"  Total CSV files        : {len(train_files)}")
    print(f"  NaN check              : {'⚠️  NaN values present' if has_nan else '✅ No NaN values'}")

    label_counts = (
        full_train_df.groupby("file_id")["label"]
        .first()
        .value_counts()
        .sort_index()
    )

    print(f"\n  Label consistency (sequence-to-one): "
          f"{'✅ Consistent' if is_label_consistent else '⚠️  Inconsistent — some CSVs have mixed labels'}")

    print("\n  Class distribution (per-sequence):")
    total_seq = label_counts.sum()
    for lbl, count in label_counts.items():
        name = LABEL_NAMES.get(lbl, str(lbl))
        print(f"    {name:20s}: {count:5d} ({count / total_seq * 100:.2f}%)")

    features    = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
    stats_df    = full_train_df[features].describe().T
    print("\n  Feature global statistics:")
    print(stats_df[["mean", "std", "min", "max"]].to_string())

    # ---- Plot: label distribution ----
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis", ax=ax)
    for p in ax.patches:
        h   = p.get_height()
        pct = h / total_seq * 100
        ax.annotate(
            f"{pct:.2f}%",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            xytext=(0, 5), textcoords="offset points",
        )
    ax.set_title("Sequence Label Distribution with Percentage (Train)", fontsize=14, pad=15)
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of Sequences (Files)")
    ax.set_ylim(0, label_counts.max() * 1.15)
    fig.tight_layout()
    save_plot(fig, "01_label_distribution.png")

    return full_train_df


# ============================================================
# MODULE 2: Deep EDA — t-SNE, Random Forest importance, user heterogeneity
# (from eda_deep.py)
# ============================================================

def load_sampled_features(
    all_csvs: list[str],
    samples_per_class: int = SAMPLES_PER_CLASS,
    n_classes: int = 6,
) -> pd.DataFrame:
    """
    Uniformly sample `samples_per_class` sequences per class (0..n_classes-1).
    Returns a per-sequence feature DataFrame with columns:
        mean_x, mean_y, mean_z, std_x, std_y, std_z,
        mean_mag, std_mag, label, user_id
    """
    files = list(all_csvs)
    random.shuffle(files)

    data_list    = []
    class_counts = {i: 0 for i in range(n_classes)}
    feature_src  = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

    for file in tqdm(files, desc="Sampling features (deep EDA)"):
        if all(c >= samples_per_class for c in class_counts.values()):
            break
        try:
            user_id = os.path.basename(os.path.dirname(file))
            df      = pd.read_csv(file)
            label   = int(df["label"].iloc[0])

            if label not in class_counts or class_counts[label] >= samples_per_class:
                continue

            feats               = df[feature_src].mean().to_dict()
            feats["mean_mag"]   = np.sqrt(feats["mean_x"] ** 2 + feats["mean_y"] ** 2 + feats["mean_z"] ** 2)
            feats["std_mag"]    = np.sqrt(feats["std_x"]  ** 2 + feats["std_y"]  ** 2 + feats["std_z"]  ** 2)
            feats["label"]      = label
            feats["user_id"]    = user_id
            data_list.append(feats)
            class_counts[label] += 1
        except Exception:
            continue

    df_feat = pd.DataFrame(data_list)
    print("\n  Sampled class distribution:")
    print(df_feat["label"].value_counts().sort_index().to_string())
    return df_feat


def run_deep_eda(all_csvs: list[str]) -> None:
    """
    Deep global mining:
      1. t-SNE manifold projection (all 6 classes)
      2. Random Forest feature importance
      3. User heterogeneity (mean_mag distribution across users)
    """
    print("\n" + "=" * 60)
    print("MODULE 2: Deep EDA — t-SNE, RF Importance, User Heterogeneity")
    print("=" * 60)

    df = load_sampled_features(all_csvs)

    feature_cols = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mean_mag", "std_mag"]
    X = df[feature_cols].values
    y = df["label"].values

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Unified HAR Deep EDA", fontsize=20, fontweight="bold")

    # --- 1. t-SNE ---
    print("  ⏳ Running t-SNE …")
    ax1      = fig.add_subplot(2, 2, 1)
    X_scaled = StandardScaler().fit_transform(X)
    tsne     = TSNE(n_components=2, perplexity=40, random_state=RANDOM_SEED, n_jobs=-1)
    X_tsne   = tsne.fit_transform(X_scaled)
    df["tsne_1"] = X_tsne[:, 0]
    df["tsne_2"] = X_tsne[:, 1]
    sns.scatterplot(
        data=df, x="tsne_1", y="tsne_2", hue="label",
        palette="tab10", alpha=0.7, s=20, ax=ax1, edgecolor=None,
    )
    ax1.set_title("1. t-SNE: Global 6-Class Manifold", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- 2. RF Feature Importance ---
    print("  ⏳ Training Random Forest for feature importance …")
    ax2 = fig.add_subplot(2, 2, 2)
    rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)
    importances      = rf.feature_importances_
    indices          = np.argsort(importances)[::-1]
    sorted_features  = [feature_cols[i] for i in indices]
    sns.barplot(x=importances[indices], y=sorted_features, palette="viridis", ax=ax2)
    ax2.set_title("2. RF Feature Importance (All Classes)", fontweight="bold")
    ax2.set_xlabel("Gini Importance")

    # --- 3. User Heterogeneity ---
    print("  ⏳ Plotting user heterogeneity …")
    ax3       = fig.add_subplot(2, 1, 2)
    top_users = sorted(df["user_id"].unique())[:15]
    df_users  = df[df["user_id"].isin(top_users)]
    sns.boxplot(
        data=df_users, x="user_id", y="mean_mag", hue="label",
        palette="tab10", ax=ax3, fliersize=2,
    )
    ax3.set_title(
        "3. User Heterogeneity: 'mean_mag' Distribution across Users",
        fontweight="bold",
    )
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "02_deep_eda_grand_unified.png")


# ============================================================
# MODULE 3: Quantitative Mining Report
# (from eda_deep_v2.py)
# ============================================================

def load_all_features(all_csvs: list[str]) -> pd.DataFrame:
    """
    Load ALL CSV files and compute per-sequence feature vectors.
    (Full scan — no sampling cap.)
    """
    data_list   = []
    feature_src = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

    for file in tqdm(all_csvs, desc="Loading all features (quantitative report)"):
        try:
            user_id = os.path.basename(os.path.dirname(file))
            df      = pd.read_csv(file)
            label   = int(df["label"].iloc[0])

            feats               = df[feature_src].mean().to_dict()
            feats["mean_mag"]   = np.sqrt(feats["mean_x"] ** 2 + feats["mean_y"] ** 2 + feats["mean_z"] ** 2)
            feats["std_mag"]    = np.sqrt(feats["std_x"]  ** 2 + feats["std_y"]  ** 2 + feats["std_z"]  ** 2)
            feats["label"]      = label
            feats["user_id"]    = user_id
            data_list.append(feats)
        except Exception:
            continue

    return pd.DataFrame(data_list)


def run_quantitative_report(all_csvs: list[str]) -> None:
    """
    Generate a quantitative mining text report covering:
      1. RF Feature Importance (exact Gini values)
      2. Feature Collinearity (Pearson, threshold |r| > 0.7)
      3. Cross-User Covariate Shift (Coefficient of Variation per label)

    Outputs:
      reports/quantitative_mining_report.txt
      reports/feature_importance.csv
      reports/feature_correlation_matrix.csv
    """
    print("\n" + "=" * 60)
    print("MODULE 3: Quantitative Mining Report")
    print("=" * 60)

    df           = load_all_features(all_csvs)
    feature_cols = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mean_mag", "std_mag"]
    X            = df[feature_cols]
    y            = df["label"]

    report_lines = ["=== HAR Quantitative Mining Report ===\n"]

    # 1. RF Feature Importance
    report_lines.append("[1. Feature Importance (Random Forest Gini Index)]")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    print("  ⏳ Training RF for full-dataset feature importance …")
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    imp_rows    = []
    for i in indices:
        line = f"  {feature_cols[i]:<15}: {importances[i]:.6f}"
        report_lines.append(line)
        imp_rows.append({"feature": feature_cols[i], "gini_importance": round(importances[i], 6)})
    report_lines.append("")

    # Save importance CSV
    imp_df = pd.DataFrame(imp_rows)
    imp_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  📄 Saved → {imp_path}")

    # 2. Feature Collinearity
    report_lines.append("[2. Feature Collinearity (|Pearson r| > 0.7)]")
    corr_matrix  = df[feature_cols].corr()
    high_corr    = False
    corr_records = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:
                report_lines.append(f"  {feature_cols[i]} & {feature_cols[j]}: {val:.4f}")
                corr_records.append({"feature_a": feature_cols[i], "feature_b": feature_cols[j], "pearson_r": round(val, 4)})
                high_corr = True
    if not high_corr:
        report_lines.append("  No highly correlated feature pairs found.")
    report_lines.append("")

    # Save correlation matrix CSV
    corr_path = os.path.join(REPORTS_DIR, "feature_correlation_matrix.csv")
    corr_matrix.round(4).to_csv(corr_path)
    print(f"  📄 Saved → {corr_path}")

    if corr_records:
        high_corr_path = os.path.join(REPORTS_DIR, "high_correlation_pairs.csv")
        pd.DataFrame(corr_records).to_csv(high_corr_path, index=False)
        print(f"  📄 Saved → {high_corr_path}")

    # 3. Cross-User Covariate Shift
    report_lines.append("[3. Cross-User Covariate Shift (Coefficient of Variation)]")
    report_lines.append("  Metric: std/mean of per-user class averages. Higher = more user bias.\n")
    cv_records = []
    for label in sorted(df["label"].unique()):
        report_lines.append(f"  --- Label {label} ({LABEL_NAMES.get(label, '')}) ---")
        df_lbl    = df[df["label"] == label]
        user_means = df_lbl.groupby("user_id")[feature_cols].mean()
        cv = (user_means.std() / user_means.mean().abs()).replace([np.inf, -np.inf], np.nan).fillna(0)
        cv_sorted = cv.sort_values(ascending=False)
        for feat, val in cv_sorted.head(3).items():
            report_lines.append(f"    {feat:<12} CV: {val:.4f}")
            cv_records.append({"label": label, "feature": feat, "cv": round(val, 4)})
        report_lines.append("")

    # Save CV CSV
    cv_path = os.path.join(REPORTS_DIR, "covariate_shift_cv.csv")
    pd.DataFrame(cv_records).to_csv(cv_path, index=False)
    print(f"  📄 Saved → {cv_path}")

    # Write text report
    report_text  = "\n".join(report_lines)
    report_fpath = os.path.join(REPORTS_DIR, "quantitative_mining_report.txt")
    with open(report_fpath, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  📄 Saved text report → {report_fpath}")
    print("\n" + report_text)


# ============================================================
# MODULE 4: Feature Space EDA (PCA + KDE)
# (from eda_space.py) — extended to configurable binary pair
# ============================================================

def extract_space_features(df: pd.DataFrame) -> dict:
    """
    Condense a 300×6 time-series into a 1-D statistical feature vector
    capturing motion intensity and postural variance.
    """
    mean_mag = np.sqrt(df["mean_x"] ** 2 + df["mean_y"] ** 2 + df["mean_z"] ** 2)
    std_mag  = np.sqrt(df["std_x"]  ** 2 + df["std_y"]  ** 2 + df["std_z"]  ** 2)
    return {
        "std_mag_mean" : std_mag.mean(),
        "std_mag_max"  : std_mag.max(),
        "std_mag_std"  : std_mag.std(),
        "std_mag_skew" : float(skew(std_mag.dropna())),
        "mean_mag_mean": mean_mag.mean(),
        "mean_mag_std" : mean_mag.std(),
    }


def run_feature_space_eda(
    all_csvs: list[str],
    label_a: int = 1,
    label_b: int = 2,
    max_samples: int = MAX_BINARY_SAMPLES,
) -> None:
    """
    Binary-class feature space analysis via KDE + PCA.
    Default: label 1 (Running) vs label 2 (Stairs Up).

    NOTE: This is intentionally binary because it visualizes the decision
    boundary between two specific activity pairs. To compare other pairs,
    change label_a / label_b.
    """
    print("\n" + "=" * 60)
    print(f"MODULE 4: Feature Space EDA — Label {label_a} vs {label_b}")
    print("=" * 60)

    files = list(all_csvs)
    random.shuffle(files)

    name_a    = LABEL_NAMES.get(label_a, f"L{label_a}")
    name_b    = LABEL_NAMES.get(label_b, f"L{label_b}")
    records   = []
    count_a   = 0
    count_b   = 0

    for file in tqdm(files, desc="Extracting space features"):
        if count_a >= max_samples and count_b >= max_samples:
            break
        try:
            df    = pd.read_csv(file)
            label = df["label"].iloc[0]
            if label == label_a and count_a < max_samples:
                rec          = extract_space_features(df)
                rec["label"] = name_a
                records.append(rec)
                count_a += 1
            elif label == label_b and count_b < max_samples:
                rec          = extract_space_features(df)
                rec["label"] = name_b
                records.append(rec)
                count_b += 1
        except Exception:
            continue

    df_feat = pd.DataFrame(records)
    print(f"  Collected: {count_a} × {name_a}, {count_b} × {name_b}")

    # ---- Plot ----
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3)
    fig.suptitle(
        f"Feature Space Analysis: {name_a} vs {name_b}",
        fontsize=16, fontweight="bold",
    )

    kde_cols   = ["std_mag_mean", "std_mag_max", "std_mag_std", "mean_mag_std"]
    kde_titles = [
        "Average Intensity (std_mag_mean)",
        "Peak Intensity (std_mag_max)",
        "Intensity Volatility (std_mag_std)",
        "Posture Variance (mean_mag_std)",
    ]
    palette = ["tab:blue", "tab:red"]

    for i, (col, title) in enumerate(zip(kde_cols, kde_titles)):
        ax = fig.add_subplot(gs[0, i] if i < 3 else gs[1, 0])
        sns.kdeplot(
            data=df_feat, x=col, hue="label",
            fill=True, common_norm=False, palette=palette, ax=ax, alpha=0.5,
        )
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel("")

    # PCA
    print("  ⏳ Running PCA …")
    ax_pca  = fig.add_subplot(gs[1, 1:])
    X_raw   = df_feat.drop("label", axis=1).values
    X_sc    = StandardScaler().fit_transform(X_raw)
    pca     = PCA(n_components=2)
    X_pca   = pca.fit_transform(X_sc)
    df_feat["PCA_1"] = X_pca[:, 0]
    df_feat["PCA_2"] = X_pca[:, 1]
    var      = pca.explained_variance_ratio_
    sns.scatterplot(
        data=df_feat, x="PCA_1", y="PCA_2", hue="label",
        palette=palette, alpha=0.6, s=25, ax=ax_pca, edgecolor=None,
    )
    ax_pca.set_title(
        f"PCA Projection (explains {sum(var)*100:.1f}% variance)", fontsize=13, fontweight="bold"
    )
    ax_pca.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax_pca.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_plot(fig, f"03_feature_space_L{label_a}_vs_L{label_b}.png")


# ============================================================
# MODULE 5: 3D Spatial Geometry (Eigenvalue Analysis)
# (from eda_geo.py) — extended to configurable binary pair
# ============================================================

def extract_eigen_features(df: pd.DataFrame) -> dict:
    """
    Compute 3-D covariance-matrix eigenvalue ratios for mean (posture) and
    std (micro-vibration) channels. These features are rotation-invariant.
    """
    def _eig_ratios(cols: list[str]) -> np.ndarray:
        data    = df[cols].values
        cov     = np.cov(data, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1].astype(float)
        total   = eigvals.sum() + 1e-8
        return eigvals / total

    mean_ratios = _eig_ratios(["mean_x", "mean_y", "mean_z"])
    std_ratios  = _eig_ratios(["std_x",  "std_y",  "std_z"])

    return {
        "mean_PC1_ratio": mean_ratios[0],
        "mean_PC2_ratio": mean_ratios[1],
        "mean_PC3_ratio": mean_ratios[2],
        "std_PC1_ratio" : std_ratios[0],
        "std_PC2_ratio" : std_ratios[1],
        "std_PC3_ratio" : std_ratios[2],
    }


def run_spatial_eda(
    all_csvs: list[str],
    label_a: int = 1,
    label_b: int = 2,
    max_samples: int = MAX_BINARY_SAMPLES,
) -> None:
    """
    3D spatial geometry analysis using covariance-matrix eigenvalue ratios.
    Default: label 1 (Running) vs label 2 (Stairs Up).
    """
    print("\n" + "=" * 60)
    print(f"MODULE 5: Spatial Geometry EDA — Label {label_a} vs {label_b}")
    print("=" * 60)

    files = list(all_csvs)
    np.random.shuffle(files)

    name_a  = LABEL_NAMES.get(label_a, f"L{label_a}")
    name_b  = LABEL_NAMES.get(label_b, f"L{label_b}")
    records = []
    count_a = 0
    count_b = 0

    for file in tqdm(files, desc="Computing eigen features"):
        if count_a >= max_samples and count_b >= max_samples:
            break
        try:
            df    = pd.read_csv(file, usecols=["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "label"])
            label = df["label"].iloc[0]
            if label == label_a and count_a < max_samples:
                feats          = extract_eigen_features(df)
                feats["label"] = name_a
                records.append(feats)
                count_a += 1
            elif label == label_b and count_b < max_samples:
                feats          = extract_eigen_features(df)
                feats["label"] = name_b
                records.append(feats)
                count_b += 1
        except Exception:
            continue

    df_feat = pd.DataFrame(records)
    print(f"  Collected: {count_a} × {name_a}, {count_b} × {name_b}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"3D Spatial Geometry (Eigenvalue Ratios): {name_a} vs {name_b}",
        fontsize=16, fontweight="bold",
    )
    palette   = ["tab:blue", "tab:red"]
    plot_vars = [
        ("mean_PC1_ratio", "Mean: Principal Axis Energy",  axes[0, 0]),
        ("mean_PC2_ratio", "Mean: Secondary Axis Energy",  axes[0, 1]),
        ("mean_PC3_ratio", "Mean: Tertiary Axis Energy",   axes[0, 2]),
        ("std_PC1_ratio",  "Std: Principal Axis Energy",   axes[1, 0]),
        ("std_PC2_ratio",  "Std: Secondary Axis Energy",   axes[1, 1]),
        ("std_PC3_ratio",  "Std: Tertiary Axis Energy",    axes[1, 2]),
    ]
    for col, title, ax in plot_vars:
        sns.kdeplot(
            data=df_feat, x=col, hue="label",
            fill=True, common_norm=False, palette=palette, ax=ax, alpha=0.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, f"04_spatial_geometry_L{label_a}_vs_L{label_b}.png")


# ============================================================
# MODULE 6: Global ACF Analysis
# (from eda_acf.py) — extended to configurable binary pair
# ============================================================

def run_acf_analysis(
    all_csvs: list[str],
    label_a: int = 1,
    label_b: int = 2,
    nlags: int = ACF_NLAGS,
) -> None:
    """
    Compute and plot the mean autocorrelation function (ACF) of the
    acceleration magnitude for two activity classes, with 10th–90th
    percentile shading to capture inter-sequence variability.

    Default: label 1 (Running) vs label 2 (Stairs Up).
    """
    print("\n" + "=" * 60)
    print(f"MODULE 6: Global ACF Analysis — Label {label_a} vs {label_b}")
    print("=" * 60)

    use_cols = ["mean_x", "mean_y", "mean_z", "label"]
    name_a   = LABEL_NAMES.get(label_a, f"L{label_a}")
    name_b   = LABEL_NAMES.get(label_b, f"L{label_b}")
    acfs_a   = []
    acfs_b   = []

    for file in tqdm(all_csvs, desc="Computing ACF"):
        try:
            df    = pd.read_csv(file, usecols=use_cols)
            label = df["label"].iloc[0]
            if label not in (label_a, label_b):
                continue
            mag        = np.sqrt(df["mean_x"] ** 2 + df["mean_y"] ** 2 + df["mean_z"] ** 2)
            acf_values = acf(mag, nlags=nlags, fft=True)
            if label == label_a:
                acfs_a.append(acf_values)
            else:
                acfs_b.append(acf_values)
        except Exception:
            continue

    print(f"  Collected: {len(acfs_a)} ACF series for {name_a}, {len(acfs_b)} for {name_b}")

    lags = np.arange(nlags + 1)

    def _plot_acf_band(ax: plt.Axes, data: list, title: str, color: str) -> None:
        arr     = np.array(data)
        mean    = arr.mean(axis=0)
        p10     = np.percentile(arr, 10, axis=0)
        p90     = np.percentile(arr, 90, axis=0)
        ax.plot(lags, mean, color=color, linewidth=2, label="Mean ACF")
        ax.fill_between(lags, p10, p90, color=color, alpha=0.3, label="P10–P90 range")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Autocorrelation")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    _plot_acf_band(axes[0], acfs_a, f"{name_a} — Global ACF (n={len(acfs_a)})", "tab:blue")
    _plot_acf_band(axes[1], acfs_b, f"{name_b} — Global ACF (n={len(acfs_b)})", "tab:red")
    axes[1].set_xlabel("Lag (timesteps)")
    fig.suptitle(
        f"Global ACF Distribution: {name_a} vs {name_b}",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    save_plot(fig, f"05_global_acf_L{label_a}_vs_L{label_b}.png")


# ============================================================
# MODULE 7: Global Statistics Summary Report
# ============================================================

def run_statistics_summary(full_train_df: pd.DataFrame) -> None:
    """
    Export structured summary tables from the full training DataFrame:
      - reports/global_feature_stats.csv   — describe() per feature
      - reports/class_feature_stats.csv    — per-class mean/std for each feature
      - reports/missing_value_report.csv   — per-column NaN counts
      - reports/full_data_summary.md       — Markdown summary report
    """
    print("\n" + "=" * 60)
    print("MODULE 7: Global Statistics Summary Reports")
    print("=" * 60)

    features = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

    # 1. Global feature stats
    global_stats = full_train_df[features].describe().T
    gs_path = os.path.join(REPORTS_DIR, "global_feature_stats.csv")
    global_stats.to_csv(gs_path)
    print(f"  📄 Saved → {gs_path}")

    # 2. Per-class feature stats
    class_stats = full_train_df.groupby("label")[features].agg(["mean", "std"])
    class_stats.columns = ["_".join(c) for c in class_stats.columns]
    cs_path = os.path.join(REPORTS_DIR, "class_feature_stats.csv")
    class_stats.to_csv(cs_path)
    print(f"  📄 Saved → {cs_path}")

    # 3. Missing value report
    nan_counts  = full_train_df.isnull().sum().reset_index()
    nan_counts.columns = ["column", "nan_count"]
    nan_path = os.path.join(REPORTS_DIR, "missing_value_report.csv")
    nan_counts.to_csv(nan_path, index=False)
    print(f"  📄 Saved → {nan_path}")

    # 4. Markdown summary report
    seq_per_label = (
        full_train_df.groupby("file_id")["label"]
        .first()
        .value_counts()
        .sort_index()
    )
    total_files = seq_per_label.sum()
    md_lines    = [
        "# HAR EDA Summary Report",
        "",
        "## Dataset Overview",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total timesteps | {len(full_train_df):,} |",
        f"| Total sequences (CSV files) | {total_files} |",
        f"| Number of features | {len(features)} |",
        f"| NaN values present | {full_train_df.isnull().values.any()} |",
        "",
        "## Class Distribution",
        "| Label | Name | Count | Pct |",
        "|-------|------|-------|-----|",
    ]
    for lbl, cnt in seq_per_label.items():
        md_lines.append(f"| {lbl} | {LABEL_NAMES.get(lbl, '')} | {cnt} | {cnt/total_files*100:.2f}% |")

    md_lines += [
        "",
        "## Feature Global Statistics",
        global_stats[["mean", "std", "min", "max"]].to_markdown(),
        "",
        "## Files Generated",
        "- `plots/01_label_distribution.png`",
        "- `plots/02_deep_eda_grand_unified.png`",
        "- `plots/03_feature_space_L1_vs_L2.png`",
        "- `plots/04_spatial_geometry_L1_vs_L2.png`",
        "- `plots/05_global_acf_L1_vs_L2.png`",
        "- `reports/quantitative_mining_report.txt`",
        "- `reports/feature_importance.csv`",
        "- `reports/feature_correlation_matrix.csv`",
        "- `reports/global_feature_stats.csv`",
        "- `reports/class_feature_stats.csv`",
        "- `reports/missing_value_report.csv`",
        "- `reports/full_data_summary.md`",
    ]

    md_path = os.path.join(REPORTS_DIR, "full_data_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  📄 Saved → {md_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 60)
    print("  HAR Integrated EDA Pipeline")
    print("=" * 60)

    ensure_dirs(PLOTS_DIR, REPORTS_DIR)

    # Scan for train CSV files (test/ is intentionally excluded — no labels)
    all_train_csvs = get_all_csvs(TRAIN_DIR)
    print(f"\n  Found {len(all_train_csvs):,} training CSVs under '{TRAIN_DIR}'")
    print(
        f"  ℹ️  NOTE: Test directory ('{TEST_DIR}') is excluded from EDA — "
        f"test CSVs have no 'label' column."
    )

    # Run all modules sequentially
    full_train_df = run_basic_eda(all_train_csvs)
    run_deep_eda(all_train_csvs)
    run_quantitative_report(all_train_csvs)
    run_feature_space_eda(all_train_csvs, label_a=1, label_b=2)
    run_spatial_eda(all_train_csvs, label_a=1, label_b=2)
    run_acf_analysis(all_train_csvs, label_a=1, label_b=2)
    run_statistics_summary(full_train_df)

    print("\n" + "=" * 60)
    print("  ✅  EDA Pipeline Complete!")
    print(f"  📁 Plots   → ./{PLOTS_DIR}/")
    print(f"  📁 Reports → ./{REPORTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
