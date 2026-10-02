import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import 6 Base Models
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

def create_time_series_data(df: pd.DataFrame, target_col='COD'):
    """
    Tạo dữ liệu cho bài toán Time Series bằng cách dịch chuyển (shift) target.
    Ví dụ: Dùng X ở thời điểm t để dự đoán y ở thời điểm t+1.
    """
    print("Đang tạo dữ liệu chuỗi thời gian (t -> t+1)...")
    df_ts = df.copy()
    
    # Tạo cột y_target bằng cách đẩy cột COD lên 1 dòng (tức là lấy giá trị của tương lai t+1)
    df_ts['y_target'] = df_ts[target_col].shift(-1)
    
    # Dòng cuối cùng sẽ bị NaN (do không có tương lai), ta phải xóa nó đi
    df_ts = df_ts.dropna(subset=['y_target'])
    return df_ts

def calculate_metrics(y_true, y_pred, n_features):
    """Tính toán 4 chỉ số đánh giá mô hình"""
    r2 = r2_score(y_true, y_pred)
    # Công thức tính Adjusted R2
    n = len(y_true)
    adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - n_features - 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'R2': r2, 'Adj_R2': adj_r2, 'RMSE': rmse, 'MAE': mae}

def build_models():
    """Định nghĩa 6 Base Models và 1 Meta Model (Stacking)"""
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0)
    }
    
    # Xây dựng Stacking Regressor từ 6 mô hình trên
    estimators = [(name, model) for name, model in models.items()]
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    # Đưa Stacking vào dict để train chung một lượt
    models['StackingEnsemble'] = stacking_model
    return models

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "water_quality_cleaned.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(DATA_PATH)
    
    # 2. Xây dựng bài toán t -> t+1
    df_ts = create_time_series_data(df, target_col='COD')
    
    # 3. Chọn Features (X) và Target (y)
    # Bỏ các cột không mang tính dự báo: Station, Timestamp, và COD hiện tại (nếu muốn dự báo khó hơn).
    # Tuy nhiên, thông thường ta vẫn giữ lại COD ở thời điểm t như một feature quan trọng.
    features = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
    X = df_ts[features]
    y = df_ts['y_target']
    
    # 4. Chia tập Train/Test (80/20) - Không shuffle vì là chuỗi thời gian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Kích thước Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 5. Chuẩn hóa Min-Max Scaling (Fit trên Train, Transform trên Test)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lưu lại scaler để web Streamlit dùng sau này
    joblib.dump(scaler, os.path.join(MODELS_DIR, "minmax_scaler.pkl"))
    
    # 6. Lấy danh sách mô hình
    models = build_models()
    
    # 7. Train & Evaluate
    results = []
    print("\nBắt đầu huấn luyện các mô hình...")
    for name, model in models.items():
        print(f"-> Đang huấn luyện {name}...")
        # Train mô hình
        model.fit(X_train_scaled, y_train)
        
        # Dự đoán
        y_pred = model.predict(X_test_scaled)
        
        # Đánh giá
        metrics = calculate_metrics(y_test, y_pred, n_features=X_test.shape[1])
        metrics['Model'] = name
        results.append(metrics)
        
        # Lưu mô hình (model weights) ra file
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
        
    # 8. Tổng hợp và lưu kết quả
    df_results = pd.DataFrame(results)
    # Sắp xếp lại thứ tự cột cho đẹp
    df_results = df_results[['Model', 'R2', 'Adj_R2', 'RMSE', 'MAE']]
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(df_results)
    
    df_results.to_csv(os.path.join(OUTPUTS_DIR, "model_metrics.csv"), index=False)
    print(f"\nĐã lưu kết quả tại {OUTPUTS_DIR}/model_metrics.csv và các mô hình tại {MODELS_DIR}/")
