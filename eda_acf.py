import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

def run_global_acf_analysis(base_path):
    print("🚀 啟動全局 ACF 效能優化掃描...")
    
    search_pattern = os.path.join(base_path, 'User_*', '*.csv')
    all_csvs = glob.glob(search_pattern)
    
    if not all_csvs:
        raise FileNotFoundError(f"在 {base_path} 中找不到任何 CSV 檔案，請檢查路徑。")
        
    print(f"📦 找到共 {len(all_csvs)} 個檔案，準備進行全局 ACF 運算...")

    l1_acfs = []
    l2_acfs = []
    
    # 針對效能優化：只讀取計算 Magnitude 和辨識 Label 必需的欄位
    use_cols = ['mean_x', 'mean_y', 'mean_z', 'label']

    for file in tqdm(all_csvs, desc="計算 ACF"):
        try:
            # 讀取輕量級資料
            df = pd.read_csv(file, usecols=use_cols)
            label = df['label'].iloc[0]
            
            # 只針對我們關心的 L1(1) 和 L2(2) 進行運算
            if label in [1, 2]:
                magnitude = np.sqrt(df['mean_x']**2 + df['mean_y']**2 + df['mean_z']**2)
                # nlags=50 產生的陣列長度會是 51 (包含 lag 0)
                acf_values = acf(magnitude, nlags=50, fft=True)
                
                if label == 1:
                    l1_acfs.append(acf_values)
                elif label == 2:
                    l2_acfs.append(acf_values)
                    
        except Exception as e:
            continue

    print(f"✅ 掃描完成！成功收集 L1: {len(l1_acfs)} 筆, L2: {len(l2_acfs)} 筆 ACF 數據。")

    # 將 List 轉換為 Numpy 2D Array 方便進行統計運算
    l1_array = np.array(l1_acfs)
    l2_array = np.array(l2_acfs)
    
    lags = np.arange(51) # X 軸 (0~50)

    # 準備畫布
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # 定義繪圖函數：畫出平均線與 10%~90% 的分佈陰影
    def plot_aggregated_acf(ax, data_array, title, color):
        mean_acf = np.mean(data_array, axis=0)
        p10_acf = np.percentile(data_array, 10, axis=0)
        p90_acf = np.percentile(data_array, 90, axis=0)
        
        ax.plot(lags, mean_acf, color=color, linewidth=2, label='Mean ACF')
        ax.fill_between(lags, p10_acf, p90_acf, color=color, alpha=0.3, label='10th - 90th Percentile Range')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Autocorrelation')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    # 繪製 L1 與 L2
    plot_aggregated_acf(axes[0], l1_array, f'L1 (Running) - Global ACF (n={len(l1_acfs)})', color='tab:blue')
    plot_aggregated_acf(axes[1], l2_array, f'L2 (Stairs Up) - Global ACF (n={len(l2_acfs)})', color='tab:red')

    axes[1].set_xlabel('Lags (Seconds)')
    plt.tight_layout()
    plt.savefig('global_acf_distribution.png', dpi=300)
    print("📈 全局分佈圖已儲存為 'global_acf_distribution.png'。")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = "train/train" 
    run_global_acf_analysis(BASE_DIR)