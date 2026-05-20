import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 設定隨機種子確保可重現
random.seed(42)
np.random.seed(42)

def extract_features(df):
    """將 300x6 的時間序列濃縮為 1D 的高階統計特徵向量"""
    # 計算 Magnitude
    mean_mag = np.sqrt(df['mean_x']**2 + df['mean_y']**2 + df['mean_z']**2)
    std_mag = np.sqrt(df['std_x']**2 + df['std_y']**2 + df['std_z']**2)
    
    features = {
        # 動態強度 (Intensity)
        'std_mag_mean': std_mag.mean(),
        'std_mag_max': std_mag.max(),
        # 強度的不穩定性 (Volatility)
        'std_mag_std': std_mag.std(),
        'std_mag_skew': skew(std_mag.dropna()),
        # 靜態重力/姿態的偏移 (Posture)
        'mean_mag_mean': mean_mag.mean(),
        'mean_mag_std': mean_mag.std()
    }
    return features

def run_feature_space_eda(base_path, max_samples=1500):
    print("🚀 啟動高階特徵空間探勘 (Feature Space EDA)...")
    
    search_pattern = os.path.join(base_path, 'User_*', '*.csv')
    all_csvs = glob.glob(search_pattern)
    random.shuffle(all_csvs)
    
    data_records = []
    l1_count, l2_count = 0, 0
    
    # 讀取與特徵萃取
    for file in tqdm(all_csvs, desc="萃取特徵"):
        if l1_count >= max_samples and l2_count >= max_samples:
            break
            
        try:
            df = pd.read_csv(file)
            label = df['label'].iloc[0]
            
            if label == 1 and l1_count < max_samples:
                features = extract_features(df)
                features['label'] = 'L1 (Running)'
                data_records.append(features)
                l1_count += 1
            elif label == 2 and l2_count < max_samples:
                features = extract_features(df)
                features['label'] = 'L2 (Stairs Up)'
                data_records.append(features)
                l2_count += 1
        except Exception:
            continue
            
    df_features = pd.DataFrame(data_records)
    print(f"\n✅ 特徵萃取完成：收集了 {l1_count} 筆 L1, {l2_count} 筆 L2 資料。")

    # ---------------- 視覺化設定 ----------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)
    
    # 1. 繪製 KDE 分佈圖 (4 個子圖)
    feature_cols = ['std_mag_mean', 'std_mag_max', 'std_mag_std', 'mean_mag_std']
    titles = [
        'Average Intensity (std_mag_mean)', 
        'Peak Intensity (std_mag_max)', 
        'Intensity Volatility (std_mag_std)',
        'Posture Variance (mean_mag_std)'
    ]
    
    for i, (col, title) in enumerate(zip(feature_cols, titles)):
        ax = fig.add_subplot(gs[0, i] if i < 3 else gs[1, 0])
        sns.kdeplot(data=df_features, x=col, hue='label', fill=True, common_norm=False, palette=['tab:blue', 'tab:red'], ax=ax, alpha=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('')

    # 2. 繪製 PCA 降維空間散佈圖
    print("⏳ 正在進行 PCA 降維運算...")
    ax_pca = fig.add_subplot(gs[1, 1:])
    
    # 標準化特徵
    X = df_features.drop('label', axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_features['PCA_1'] = X_pca[:, 0]
    df_features['PCA_2'] = X_pca[:, 1]
    
    var_ratio = pca.explained_variance_ratio_
    
    sns.scatterplot(data=df_features, x='PCA_1', y='PCA_2', hue='label', palette=['tab:blue', 'tab:red'], alpha=0.6, s=30, ax=ax_pca, edgecolor=None)
    ax_pca.set_title(f'PCA Feature Space Projection (Explains {sum(var_ratio)*100:.1f}% Variance)', fontsize=14, fontweight='bold')
    ax_pca.set_xlabel(f'Principal Component 1 ({var_ratio[0]*100:.1f}%)')
    ax_pca.set_ylabel(f'Principal Component 2 ({var_ratio[1]*100:.1f}%)')
    ax_pca.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('feature_space_analysis.png', dpi=300)
    print("📈 分析圖表已儲存為 'feature_space_analysis.png'。")
    plt.show()

if __name__ == "__main__":
    # 請確保這個路徑指向你的專案資料夾結構
    BASE_DIR = "train/train" 
    run_feature_space_eda(BASE_DIR)