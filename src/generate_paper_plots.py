import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from train_pipeline import create_time_series_data

# Đảm bảo hiển thị đẹp và rõ nét
plt.rcParams.update({'font.size': 12})
sns.set_theme(style="whitegrid")

def generate_eda_and_plots():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "water_quality_cleaned.csv")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Đọc dữ liệu
    df = pd.read_csv(DATA_PATH)
    features = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
    df_features = df[features]
    
    print("1. Đang xuất Bảng Thống kê mô tả (Descriptive Statistics)...")
    desc_stats = df_features.describe().T
    desc_stats = desc_stats[['mean', 'std', 'min', 'max', '25%', '75%']]
    desc_stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)']
    desc_stats.to_csv(os.path.join(OUTPUTS_DIR, "descriptive_statistics.csv"))
    print("-> Đã lưu outputs/descriptive_statistics.csv")
    
    print("\n2. Đang vẽ Ma trận tương quan (Pearson Correlation Heatmap)...")
    plt.figure(figsize=(10, 8))
    corr_matrix = df_features.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Pearson Correlation Heatmap (Mức độ tương quan giữa các thông số)", pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print("-> Đã lưu outputs/correlation_heatmap.png")
    
    print("\n3. Đang vẽ Biểu đồ Thực tế vs Dự báo (Actual vs Predicted) cho Stacking Ensemble...")
    df_ts = create_time_series_data(df, target_col='COD')
    X = df_ts[features]
    y = df_ts['y_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "minmax_scaler.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "models", "StackingEnsemble.pkl"))
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Vẽ 150 chu kỳ đầu tiên của tập Test cho dễ nhìn
    n_points = 150
    actual = y_test.values[:n_points]
    predicted = y_pred[:n_points]
    
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Thực tế (Actual COD)', color='royalblue', linewidth=2, marker='o', markersize=4, alpha=0.8)
    plt.plot(predicted, label='Dự báo (Predicted COD)', color='crimson', linewidth=2, linestyle='--', marker='x', markersize=4, alpha=0.9)
    
    plt.title("So sánh Thực tế và Dự báo COD (Stacking Ensemble) trên 150 chu kỳ thử nghiệm", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Thời gian (Chu kỳ 5 phút)", fontsize=12)
    plt.ylabel("Nồng độ COD (mg/L)", fontsize=12)
    plt.legend(loc="upper left", frameon=True, shadow=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    print("-> Đã lưu outputs/actual_vs_predicted.png")
    
    print("\nHoàn tất! Các file đã được lưu vào thư mục outputs/")

if __name__ == "__main__":
    generate_eda_and_plots()
