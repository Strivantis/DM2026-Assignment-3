import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def extract_eigen_features(df):
    """計算三軸訊號的 3D 共變異數矩陣特徵值 (旋轉不變性)"""
    # 針對 mean (姿勢/巨觀加速度)
    mean_data = df[['mean_x', 'mean_y', 'mean_z']].values
    cov_mean = np.cov(mean_data, rowvar=False)
    # 計算特徵值並由大到小排序
    eigvals_mean = np.sort(np.linalg.eigvals(cov_mean))[::-1]
    
    # 針對 std (微觀震動)
    std_data = df[['std_x', 'std_y', 'std_z']].values
    cov_std = np.cov(std_data, rowvar=False)
    eigvals_std = np.sort(np.linalg.eigvals(cov_std))[::-1]
    
    # 為了避免尺度問題，我們計算特徵值的「比例 (Ratio)」
    # 這代表能量分佈在主軸、次軸、第三軸的百分比
    sum_eig_mean = np.sum(eigvals_mean) + 1e-8
    sum_eig_std = np.sum(eigvals_std) + 1e-8
    
    features = {
        'mean_PC1_ratio': eigvals_mean[0] / sum_eig_mean, # 主軸能量佔比
        'mean_PC2_ratio': eigvals_mean[1] / sum_eig_mean, # 次軸能量佔比
        'mean_PC3_ratio': eigvals_mean[2] / sum_eig_mean, # 第三軸能量佔比
        'std_PC1_ratio': eigvals_std[0] / sum_eig_std,
        'std_PC2_ratio': eigvals_std[1] / sum_eig_std,
        'std_PC3_ratio': eigvals_std[2] / sum_eig_std
    }
    return features

def run_spatial_eda(base_path, max_samples=1500):
    print("🚀 啟動 3D 空間幾何特徵探勘 (Eigenvalue Analysis)...")
    
    search_pattern = os.path.join(base_path, 'User_*', '*.csv')
    all_csvs = glob.glob(search_pattern)
    np.random.seed(42)
    np.random.shuffle(all_csvs)
    
    records = []
    l1_count, l2_count = 0, 0
    
    for file in tqdm(all_csvs, desc="計算空間特徵"):
        if l1_count >= max_samples and l2_count >= max_samples:
            break
        try:
            df = pd.read_csv(file, usecols=['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'label'])
            label = df['label'].iloc[0]
            
            if label == 1 and l1_count < max_samples:
                feats = extract_eigen_features(df)
                feats['label'] = 'L1 (Running)'
                records.append(feats)
                l1_count += 1
            elif label == 2 and l2_count < max_samples:
                feats = extract_eigen_features(df)
                feats['label'] = 'L2 (Stairs Up)'
                records.append(feats)
                l2_count += 1
        except Exception:
            continue
            
    df_features = pd.DataFrame(records)
    print(f"\n✅ 分析完成：L1 共 {l1_count} 筆, L2 共 {l2_count} 筆。")

    # 繪製 KDE 分佈圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("3D Spatial Geometry Analysis (Eigenvalue Ratios of Covariance Matrix)", fontsize=16, fontweight='bold')
    
    plot_vars = [
        ('mean_PC1_ratio', 'Mean: Principal Axis Energy', axes[0,0]),
        ('mean_PC2_ratio', 'Mean: Secondary Axis Energy', axes[0,1]),
        ('mean_PC3_ratio', 'Mean: Tertiary Axis Energy', axes[0,2]),
        ('std_PC1_ratio', 'Std: Principal Axis Energy', axes[1,0]),
        ('std_PC2_ratio', 'Std: Secondary Axis Energy', axes[1,1]),
        ('std_PC3_ratio', 'Std: Tertiary Axis Energy', axes[1,2])
    ]
    
    for col, title, ax in plot_vars:
        sns.kdeplot(data=df_features, x=col, hue='label', fill=True, common_norm=False, 
                    palette=['tab:blue', 'tab:red'], ax=ax, alpha=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('spatial_geometry_analysis.png', dpi=300)
    print("📈 分析圖表已儲存為 'spatial_geometry_analysis.png'。")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = "train/train" 
    run_spatial_eda(BASE_DIR)