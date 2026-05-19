import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 參數設定 (請修改為你的實際路徑)
# ==========================================
TRAIN_DIR = './train/train'  # 包含 User_001-060 的資料夾
TEST_DIR = './data/data'     # 包含 User_061-100 的資料夾

def run_eda():
    print("🚀 啟動資料探勘與預處理分析...\n")
    
    # 1. 載入訓練集結構資訊
    train_users = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    print(f"📦 找到 {len(train_users)} 個訓練集使用者資料夾。")
    
    train_files = glob.glob(os.path.join(TRAIN_DIR, '**', '*.csv'), recursive=True)
    print(f"📄 找到共 {len(train_files)} 個訓練集 CSV 檔案。")

    # 2. 抽樣讀取並建立全局統計 DataFrame
    print("⏳ 正在讀取訓練集資料進行統計分析...")
    
    df_list = []
    file_label_mapping = {}
    is_label_consistent = True
    has_nan = False
    
    # 為了節省時間與記憶體，如果檔案超過 10000 個，我們全讀也無妨 (資料很輕量)
    for file_path in tqdm(train_files, desc="讀取 Train CSV"):
        df = pd.read_csv(file_path)
        
        # 檢查缺失值
        if df.isnull().values.any():
            has_nan = True
            
        # 檢查 CSV 內的 Label 是否一致
        unique_labels = df['label'].unique()
        file_id = df['file_id'].iloc[0]
        
        if len(unique_labels) > 1:
            is_label_consistent = False
        
        # 紀錄這個檔案的標籤 (假設一致，取第一個)
        file_label_mapping[file_id] = unique_labels[0]
        
        # 將資料加入總表以計算特徵分佈
        df_list.append(df)

    full_train_df = pd.concat(df_list, ignore_index=True)
    
    # 3. 輸出探勘報告
    print("\n" + "="*40)
    print("📊 [架構師 EDA 報告書]")
    print("="*40)
    
    print(f"1. 總資料列數 (Total Timesteps): {len(full_train_df)}")
    
    print(f"\n2. 缺失值檢查 (NaN): {'⚠️ 發現缺失值' if has_nan else '✅ 無缺失值'}")
    
    print(f"\n3. 標籤一致性檢查 (Sequence-to-One 驗證):")
    if is_label_consistent:
        print("   ✅ 完美！每個 CSV (5分鐘序列) 內的標籤完全一致。")
        print("   👉 結論：這是一個標準的多變數時間序列分類任務 (Sequence-to-One)。")
    else:
        print("   ⚠️ 注意：部分 CSV 內包含多種標籤！需要進一步處理為 Sequence-to-Sequence。")
        
    print("\n4. 類別分佈 (Class Distribution):")
    label_counts = full_train_df.groupby('file_id')['label'].first().value_counts().sort_index()
    for lbl, count in label_counts.items():
        print(f"   - Label {lbl}: {count} 個序列 ({count/len(train_files)*100:.2f}%)")
        
    print("\n5. 特徵全域統計量 (Global Mean & Std for Normalization):")
    features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    stats_df = full_train_df[features].describe().T
    print(stats_df[['mean', 'std', 'min', 'max']])
    
    # 繪製類別分佈圖
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Sequence Label Distribution (Train)')
    plt.xlabel('Label')
    plt.ylabel('Number of Sequences (Files)')
    plt.savefig('label_distribution.png')
    print("\n📈 已生成類別分佈圖：'label_distribution.png'")
    print("="*40)

if __name__ == "__main__":
    run_eda()