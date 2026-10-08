import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from train_pipeline import create_time_series_data

# Cấu hình biểu đồ
plt.rcParams.update({'font.size': 12})
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "water_quality_cleaned.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Đọc dữ liệu và load model
df = pd.read_csv(DATA_PATH)
features = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
df_ts = create_time_series_data(df, target_col='COD')
X = df_ts[features]
y = df_ts['y_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = joblib.load(os.path.join(BASE_DIR, "models", "minmax_scaler.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "models", "StackingEnsemble.pkl"))

# Dự đoán và tính phần dư
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
residuals = y_test - y_pred

# Vẽ biểu đồ 1x2 (Histogram và Scatter)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Biểu đồ 1: Histogram phân phối phần dư
sns.histplot(residuals, kde=True, ax=axes[0], color='purple')
axes[0].set_title('Phân phối Phần dư (Residual Distribution)', fontweight='bold')
axes[0].set_xlabel('Phần dư (Thực tế - Dự báo)')
axes[0].set_ylabel('Tần suất')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)

# Biểu đồ 2: Residuals vs Fitted
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], color='teal', alpha=0.6)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Phần dư vs Giá trị Dự báo (Residuals vs. Fitted)', fontweight='bold')
axes[1].set_xlabel('Giá trị Dự báo (Predicted COD)')
axes[1].set_ylabel('Phần dư (Residuals)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "residual_plots.png"), dpi=300)
print(f"Hoàn tất! Đã lưu ảnh tại thư mục outputs/residual_plots.png")
