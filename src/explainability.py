import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import hàm tạo dữ liệu chuỗi thời gian từ file train_pipeline
from train_pipeline import create_time_series_data

def load_data_and_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "water_quality_cleaned.csv")
    
    # SHAP TreeExplainer hoạt động hoàn hảo với các mô hình cây. 
    # Do ExtraTrees có R2 rất cao (0.899) gần bằng Stacking (0.900), ta sẽ dùng nó để đại diện giải thích.
    MODEL_PATH = os.path.join(BASE_DIR, "models", "ExtraTrees.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "minmax_scaler.pkl")
    
    df = pd.read_csv(DATA_PATH)
    df_ts = create_time_series_data(df, target_col='COD')
    
    features = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
    X = df_ts[features]
    y = df_ts['y_target']
    
    # Lấy tập Test giống như lúc train
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Load model và scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Scale tập X_test
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
    
    return model, X_test_scaled_df, features

def generate_shap_plots():
    print("Đang chuẩn bị dữ liệu và mô hình để tính toán SHAP...")
    model, X_test_scaled, feature_names = load_data_and_model()
    
    OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    print("Đang tính toán giá trị SHAP (TreeExplainer)...")
    # TreeExplainer: Thuật toán giải thích dành riêng cho các mô hình dạng Cây Quyết định
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # 1. Vẽ SHAP Bar Plot (Biểu đồ cột - Mức độ quan trọng trung bình của từng đặc trưng)
    print("- Đang tạo biểu đồ Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Độ quan trọng của các thông số (Feature Importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Vẽ SHAP Summary Plot (Biểu đồ phân tán - Tác động cục bộ và toàn cục)
    print("- Đang tạo biểu đồ Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nTuyệt vời! Các biểu đồ SHAP đã được xuất thành công vào thư mục: {OUTPUTS_DIR}")

if __name__ == "__main__":
    # Đảm bảo matplotlib không bị lỗi hiển thị font
    plt.rcParams.update({'font.size': 12})
    generate_shap_plots()
