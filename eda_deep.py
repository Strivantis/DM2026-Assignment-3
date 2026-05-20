import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

# 固定亂數種子
random.seed(42)
np.random.seed(42)

def load_global_data(base_path, samples_per_class=800):
    """均勻抽樣所有 6 個類別，並保留 User 資訊"""
    print("📦 正在載入全域資料...")
    search_pattern = os.path.join(base_path, 'User_*', '*.csv')
    all_csvs = glob.glob(search_pattern)
    random.shuffle(all_csvs)
    
    # 儲存資料
    data_list = []
    class_counts = {i: 0 for i in range(6)}
    
    for file in tqdm(all_csvs, desc="讀取 CSV"):
        # 檢查是否已滿足所有類別的抽樣數量
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
        try:
            # 讀取 User ID (從資料夾名稱提取，例如 User_001)
            user_id = file.split(os.sep)[-2] 
            
            df = pd.read_csv(file)
            label = int(df['label'].iloc[0])
            
            if class_counts[label] < samples_per_class:
                # 萃取單一序列的統計平均作為特徵
                features = df[['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']].mean().to_dict()
                
                # 計算 Magnitude
                features['mean_mag'] = np.sqrt(features['mean_x']**2 + features['mean_y']**2 + features['mean_z']**2)
                features['std_mag'] = np.sqrt(features['std_x']**2 + features['std_y']**2 + features['std_z']**2)
                
                features['label'] = label
                features['user_id'] = user_id
                data_list.append(features)
                class_counts[label] += 1
        except Exception:
            continue
            
    df_all = pd.DataFrame(data_list)
    print("\n✅ 資料載入完成，各類別分佈：")
    print(df_all['label'].value_counts().sort_index())
    return df_all

def run_deep_mining(base_path):
    df = load_global_data(base_path)
    
    feature_cols = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'mean_mag', 'std_mag']
    X = df[feature_cols]
    y = df['label']
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Grand Unified HAR Data Mining", fontsize=20, fontweight='bold')
    
    # ==========================================
    # 1. 全類別 t-SNE 高維流形降維
    # ==========================================
    print("\n⏳ 正在執行 t-SNE 降維運算 (這可能需要 1-2 分鐘)...")
    ax1 = fig.add_subplot(2, 2, 1)
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]
    
    sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='label', 
                    palette='tab10', alpha=0.7, s=20, ax=ax1, edgecolor=None)
    ax1.set_title("1. t-SNE: Global 6-Class Manifold", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # 2. Random Forest 特徵重要性評估
    # ==========================================
    print("⏳ 正在訓練 Random Forest 評估特徵重要性...")
    ax2 = fig.add_subplot(2, 2, 2)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    
    sns.barplot(x=importances[indices], y=sorted_features, palette='viridis', ax=ax2)
    ax2.set_title("2. Random Forest Feature Importance (All Classes)", fontweight='bold')
    ax2.set_xlabel("Gini Importance")

    # ==========================================
    # 3. 使用者異質性分析 (User-Level Covariate Shift)
    # ==========================================
    print("⏳ 正在繪製使用者異質性分析...")
    ax3 = fig.add_subplot(2, 1, 2)
    
    # 挑選前 15 個 User 來觀察他們的「平均運動強度 (mean_mag)」分佈
    top_users = sorted(df['user_id'].unique())[:15]
    df_users = df[df['user_id'].isin(top_users)]
    
    sns.boxplot(data=df_users, x='user_id', y='mean_mag', hue='label', palette='tab10', ax=ax3, fliersize=2)
    ax3.set_title("3. User Heterogeneity: Distribution of 'mean_mag' across Different Users", fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('grand_unified_eda.png', dpi=300)
    print("\n🎯 大功告成！深度探勘圖表已儲存為 'grand_unified_eda.png'。")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = "train/train" 
    run_deep_mining(BASE_DIR)