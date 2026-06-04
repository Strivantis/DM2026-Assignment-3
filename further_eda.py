import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ==========================================
# 參數設定區 (可依據實際需求調整)
# ==========================================
CFG = {
    "train_root": "./train/train",
    "test_root": "./test/test",
    "sample_rate": 1.0,  # 若資料量太大，可設為 0.1 進行 10% 抽樣加速運算
    "n_components_pca": 2,
    "dbscan_eps": 0.5,   # DBSCAN 的半徑參數，需根據資料實際分佈微調
    "dbscan_min_samples": 5
}

# ==========================================
# 1. 資料讀取與特徵提取
# ==========================================
def extract_features_from_csv(file_path):
    """
    從單個 CSV 提取基本特徵 (平均值、標準差等)。
    假設 CSV 內為數值型時間序列資料 (如 accelerometer 資料)。
    """
    try:
        df = pd.read_csv(file_path)
        # 只取數值型欄位
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        
        # 簡單特徵工程：計算每個感測器通道的 Mean 和 Std
        features = {}
        for col in numeric_df.columns:
            features[f"{col}_mean"] = numeric_df[col].mean()
            features[f"{col}_std"] = numeric_df[col].std()
            # 若有需要，可以在此加入 jerk, rolling_var 等特徵
            
        return features
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_data(root_dir, dataset_type):
    """讀取資料夾下所有使用者的 CSV 並轉換為 DataFrame"""
    print(f"正在載入 {dataset_type} 資料從: {root_dir}...")
    all_files = glob.glob(os.path.join(root_dir, "*", "*.csv"))
    
    # 若設定抽樣，則隨機抽取檔案以加速
    if CFG["sample_rate"] < 1.0:
        np.random.shuffle(all_files)
        all_files = all_files[:int(len(all_files) * CFG["sample_rate"])]
        
    data_list = []
    for file in all_files:
        features = extract_features_from_csv(file)
        if features:
            features['dataset'] = dataset_type
            data_list.append(features)
            
    df = pd.DataFrame(data_list)
    print(f"成功載入 {len(df)} 筆 {dataset_type} 資料。")
    return df

# ==========================================
# 2. 量化共變異數偏移 (Covariate Shift)
# ==========================================
def analyze_covariate_shift(train_df, test_df, feature_cols):
    print("\n--- 執行共變異數偏移分析 (KS-Test) ---")
    ks_results = []
    for col in feature_cols:
        stat, p_value = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        ks_results.append({'Feature': col, 'KS_Stat': stat, 'P_Value': p_value})
        
    ks_df = pd.DataFrame(ks_results).sort_values(by='KS_Stat', ascending=False)
    print("KS 統計量最高 (偏移最嚴重) 的前 5 個特徵：")
    print(ks_df.head())
    
    # 繪製偏移最嚴重的特徵 KDE 圖
    top_feature = ks_df.iloc[0]['Feature']
    plt.figure(figsize=(8, 5))
    sns.kdeplot(train_df[top_feature], label='Train', fill=True)
    sns.kdeplot(test_df[top_feature], label='Test', fill=True)
    plt.title(f"Covariate Shift: KDE of {top_feature}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('covariate_shift.png')
    plt.close()
    print(f"已儲存 {top_feature} 的 KDE 密度圖至 covariate_shift.png")

# ==========================================
# 3. 全域流形對齊 (PCA & t-SNE)
# ==========================================
def analyze_manifold_alignment(combined_df, feature_cols):
    print("\n--- 執行全域流形對齊 (PCA & t-SNE) ---")
    X = combined_df[feature_cols].fillna(0).values
    y_dataset = combined_df['dataset'].values
    
    # 必須先進行正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    print("計算 PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE (資料量大時可能需要幾分鐘)
    print("計算 t-SNE (這可能需要一點時間)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_dataset, alpha=0.5, ax=axes[0])
    axes[0].set_title('PCA: Train vs Test')
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_dataset, alpha=0.5, ax=axes[1])
    axes[1].set_title('t-SNE: Train vs Test')
    
    plt.tight_layout()
    plt.savefig('manifold_alignment.png')
    plt.close()
    print("已儲存降維可視化圖至 manifold_alignment.png")
    
    return X_scaled # 回傳正規化後的特徵供後續使用

# ==========================================
# 4. 評估偽標籤適用性 (DBSCAN)
# ==========================================
def evaluate_pseudo_labeling(test_df, feature_cols):
    print("\n--- 評估偽標籤適用性 (Test Set 分群) ---")
    X_test = test_df[feature_cols].fillna(0).values
    
    # 正規化
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # 為了方便視覺化，我們先用 PCA 降至 2 維來做分群 (實務上可在高維度做，但需調參)
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_scaled)
    
    # 執行 DBSCAN
    dbscan = DBSCAN(eps=CFG["dbscan_eps"], min_samples=CFG["dbscan_min_samples"])
    clusters = dbscan.fit_predict(X_test_pca)
    
    # 評估分群品質 (排除 noise點 -1)
    if len(set(clusters)) > 1:
        mask = clusters != -1
        if sum(mask) > 1:
            score = silhouette_score(X_test_pca[mask], clusters[mask])
            print(f"Silhouette Score (分群輪廓係數): {score:.4f} (越接近 1 代表群集越清晰)")
    else:
        print("DBSCAN 無法找到有效的群集，可能需要調整 eps 或 min_samples 參數。")
        
    # 繪製分群結果
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
    plt.title('DBSCAN Clustering on Test Set (PCA Reduced)')
    plt.colorbar(scatter, label='Cluster Label (-1 is Noise)')
    plt.tight_layout()
    plt.savefig('dbscan_clusters.png')
    plt.close()
    print("已儲存測試集 DBSCAN 分群圖至 dbscan_clusters.png")

# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    # 1. 載入資料
    train_df = load_data(CFG["train_root"], "Train")
    test_df = load_data(CFG["test_root"], "Test")
    
    if not train_df.empty and not test_df.empty:
        # 【修改重點】取得特徵欄位名稱：只保留 Train 和 Test "都有"的欄位，並排除 'dataset'
        feature_cols = [col for col in train_df.columns if col in test_df.columns and col != 'dataset']
        
        print(f"\n將進行分析的特徵數量: {len(feature_cols)} 個")
        
        # 組合資料以便進行後續降維 (只取共有的特徵與 dataset 欄位)
        combined_df = pd.concat([
            train_df[feature_cols + ['dataset']], 
            test_df[feature_cols + ['dataset']]
        ], ignore_index=True)
        
        # 2. 分析共變異數偏移
        analyze_covariate_shift(train_df, test_df, feature_cols)
        
        # 3. 執行流形對齊
        X_scaled_combined = analyze_manifold_alignment(combined_df, feature_cols)
        
        # 4. 評估偽標籤 (僅針對 Test Set)
        evaluate_pseudo_labeling(test_df, feature_cols)
        
        print("\n資料探勘完成！請查看生成的 .png 圖片以進行決策。")
    else:
        print("資料載入失敗，請確認檔案路徑是否正確。")