import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
from tqdm import tqdm

def load_global_data_for_stats(base_path):
    search_pattern = os.path.join(base_path, 'User_*', '*.csv')
    all_csvs = glob.glob(search_pattern)
    
    data_list = []
    
    for file in tqdm(all_csvs, desc="讀取與計算全域數值"):
        try:
            user_id = file.split(os.sep)[-2]
            df = pd.read_csv(file)
            label = int(df['label'].iloc[0])
            
            # 計算基礎統計特徵
            features = df[['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']].mean().to_dict()
            features['mean_mag'] = np.sqrt(features['mean_x']**2 + features['mean_y']**2 + features['mean_z']**2)
            features['std_mag'] = np.sqrt(features['std_x']**2 + features['std_y']**2 + features['std_z']**2)
            
            features['label'] = label
            features['user_id'] = user_id
            data_list.append(features)
        except Exception:
            continue
            
    return pd.DataFrame(data_list)

def run_quantitative_mining(base_path):
    df = load_global_data_for_stats(base_path)
    feature_cols = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'mean_mag', 'std_mag']
    
    report_lines = []
    report_lines.append("=== HAR Quantitative Mining Report ===\n")
    
    # 1. Random Forest Feature Importance (Exact Values)
    report_lines.append("[1. Feature Importance (Random Forest Gini Index)]")
    X = df[feature_cols]
    y = df['label']
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        report_lines.append(f"{feature_cols[i]:<15}: {importances[i]:.6f}")
    report_lines.append("\n")
    
    # 2. Feature Collinearity (Pearson Correlation Matrix)
    report_lines.append("[2. Feature Collinearity (Correlation > 0.7 or < -0.7)]")
    corr_matrix = df[feature_cols].corr()
    high_corr_found = False
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:
                report_lines.append(f"{feature_cols[i]} & {feature_cols[j]}: {val:.4f}")
                high_corr_found = True
    if not high_corr_found:
        report_lines.append("No highly correlated feature pairs found.")
    report_lines.append("\n")
    
    # 3. User-Level Covariate Shift Quantification
    report_lines.append("[3. Cross-User Covariate Shift (Coefficient of Variation)]")
    report_lines.append("Metric: Std/Mean of the class averages across different users. Higher value = Higher user bias.\n")
    
    for label in sorted(df['label'].unique()):
        report_lines.append(f"--- Label {label} ---")
        df_label = df[df['label'] == label]
        
        # 計算每個 user 在該 label 下的平均特徵
        user_means = df_label.groupby('user_id')[feature_cols].mean()
        
        # 計算 user 間的變異係數 (CV = std / mean)
        global_mean = user_means.mean()
        global_std = user_means.std()
        cv = (global_std / abs(global_mean)).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 排序並列出 CV 最大的前三個特徵
        cv_sorted = cv.sort_values(ascending=False)
        for feat, val in cv_sorted.head(3).items():
            report_lines.append(f"  {feat:<10} CV: {val:.4f}")
    
    # 輸出報告
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open("quantitative_mining_report.txt", "w") as f:
        f.write(report_text)
    print("\n✅ 報告已儲存為 quantitative_mining_report.txt")

if __name__ == "__main__":
    BASE_DIR = "train/train" 
    run_quantitative_mining(BASE_DIR)