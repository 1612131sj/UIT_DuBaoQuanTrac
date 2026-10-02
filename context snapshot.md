# TÓM TẮT NGỮ CẢNH HỆ THỐNG / DỰ ÁN

## 1. Tổng quan & Mục tiêu dự án
- **Môn học**: Học máy tiên tiến và ứng dụng (UIT)
- **Chủ đề**: Dự báo chỉ số chất lượng nước sử dụng mô hình học máy kết hợp (Stacked Regression Ensemble) và giải thích mô hình bằng SHAP trên dữ liệu trạm quan trắc tự động.
- **Bài báo tham khảo**: Rakesh Choudhary et al. (2025), "Predicting water quality index using stacked ensemble regression and SHAP based explainable artificial intelligence", Nature Scientific Reports, 15:31139.
- **Mục tiêu**: Xây dựng một dự án Học máy hoàn chỉnh, có cấu trúc module để dự báo các chỉ số ô nhiễm nước liên tục (ví dụ: COD hoặc WQI tại bước t+1) từ dữ liệu thô của các trạm quan trắc môi trường tự động. Đánh giá hiệu suất định lượng, giải thích các dự đoán sử dụng SHAP và cung cấp giao diện người dùng tương tác bằng Streamlit để demo.

## 2. Đặc tả dữ liệu & Đặc trưng
- **Dữ liệu đầu vào**: Nhiều file log thô (`.txt`) từ các trạm quan trắc chất lượng nước liên tục (chứa các thông số: `pH`, `Temp` (Nhiệt độ), `TSS`, `COD`, `NH4+`, `Color` (Độ màu), `Flowin`/`Flow2`, `Flowout`/`Flow1` kèm theo thời gian).
- **Biến mục tiêu (\(y\))**: `COD` (mg/L) ở thời điểm tiếp theo hoặc Chỉ số chất lượng nước (`WQI`) được tính toán.
- **Biến đặc trưng (\(X\))**: Các thông số hóa lý & thủy lực: `pH`, `Temp`, `Color`, `TSS`, `NH4+`, `Flowin` (hoặc lưu lượng kênh).
- **Quy trình tiền xử lý dữ liệu**:
  1. Tự động gộp tất cả các file thô `.txt` trong thư mục data thành một Pandas DataFrame duy nhất.
  2. Điền dữ liệu bị thiếu bằng **Phương pháp trung vị (Median Imputation)**.
  3. Lọc nhiễu (outlier) sử dụng phương pháp **Khoảng tứ phân vị (IQR)** (\(Q1 - 1.5 \times IQR\) đến \(Q3 + 1.5 \times IQR\)).
  4. Chuẩn hóa đặc trưng/Min-Max Scaling.
  5. Chia tập dữ liệu: **80% Huấn luyện / 20% Kiểm thử** với **Kiểm chứng chéo 5 lần (5-fold Cross-Validation)**.

## 3. Kiến trúc Học máy (7 Thuật toán)
- **Mô hình cơ sở (6 Mô hình hồi quy - Base-Learners)**:
  1. Random Forest Regressor
  2. Extra Trees Regressor
  3. Gradient Boosting Regressor
  4. AdaBoost Regressor
  5. XGBoost Regressor (`xgboost`)
  6. CatBoost Regressor (`catboost`)
- **Mô hình meta (1 Mô hình - Meta-Learner)**:
  - Linear Regression hoạt động bên trong mô hình **Stacking Regressor** (tổng hợp các dự đoán đã qua kiểm chứng chéo của tất cả 6 mô hình cơ sở).
- **Chỉ số đánh giá**:
  - Hệ số xác định (\(R^2\)), \(R^2\) hiệu chỉnh (Adjusted \(R^2\)), Sai số toàn phương trung bình (RMSE), Sai số tuyệt đối trung bình (MAE).
- **Trí tuệ nhân tạo có thể giải thích (XAI)**:
  - SHAP (`shap.TreeExplainer`) để phân tích mức độ quan trọng của đặc trưng toàn cục (Summary Plot) và phân tích tương tác cục bộ (Dependence Plot, Force Plot).

## 4. Cấu trúc thư mục & Kiến trúc mã nguồn
```text
water_quality_ml/
├── data/
│   ├── raw/                   # Chứa các file log .txt thô
│   └── processed/             # Dữ liệu đã làm sạch (water_quality_cleaned.csv)
├── models/                    # Lưu các mô hình đã huấn luyện (.joblib / .pkl)
├── outputs/                   # Lưu các biểu đồ SHAP & file CSV kết quả đánh giá
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Đọc .txt, gộp file, loại bỏ nhiễu IQR, điền giá trị thiếu bằng trung vị, chuẩn hóa
│   ├── train_pipeline.py      # Huấn luyện 6 mô hình cơ sở + Stacking Regressor, 5-fold CV, tính toán chỉ số
│   └── explainability.py      # Sinh biểu đồ SHAP TreeExplainer summary và force plot
├── notebooks/
│   └── EDA_and_Modeling.ipynb # Notebook Jupyter tương tác để làm EDA & huấn luyện từng bước
├── app.py                     # Ứng dụng web Streamlit để demo mô hình tương tác
├── requirements.txt           # Các thư viện cần thiết (pandas, scikit-learn, xgboost, catboost, shap, streamlit)
└── README.md                  # Tài liệu dự án & hướng dẫn cài đặt
```

## 5. Các tác vụ thực thi cho AI
1. Tạo tất cả các cấu trúc thư mục và file cần thiết (`requirements.txt`, `README.md`).
2. Viết mã Python mạnh mẽ, có cấu trúc module trong `src/data_preprocessing.py` để phân tích các file log thô `.txt` (trích xuất các thông số như `pH`, `TSS`, `COD`, `NH4+`, `Temp`, `Color`, `Flow`) và xử lý dữ liệu thiếu/nhiễu thông qua phương pháp Trung vị và IQR.
3. Viết `src/train_pipeline.py` để xây dựng, tinh chỉnh và đánh giá tất cả 6 mô hình cơ sở cùng với Stacking Ensemble sử dụng \(R^2\), Adjusted \(R^2\), RMSE và MAE. Lưu các mô hình đã huấn luyện vào thư mục `models/`.
4. Viết `src/explainability.py` để tính toán giá trị SHAP và lưu các biểu đồ trực quan.
5. Viết một file `app.py` (Streamlit) nhẹ nhàng, tinh tế cho phép người dùng điều chỉnh các thông số chất lượng nước đầu vào và xem dự đoán COD theo thời gian thực cùng với giải thích SHAP.