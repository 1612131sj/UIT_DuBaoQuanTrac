# LỘ TRÌNH THỰC HIỆN ĐỒ ÁN (Dùng cho Slide)

## 0. Tổng quan Bài toán & Ngữ cảnh
- **Ngữ cảnh nghiên cứu**: Đồ án ứng dụng phương pháp Học máy kết hợp (Stacked Ensemble Regression) và Trí tuệ nhân tạo có thể giải thích (SHAP) dựa trên nghiên cứu tham khảo. Các phương pháp này được tùy chỉnh để giải quyết bài toán thực tiễn: Dự báo và cảnh báo sớm chất lượng nước thải tại các Khu công nghiệp.
- **Mục tiêu bài toán**: Khai thác dữ liệu chuỗi thời gian từ hệ thống quan trắc tự động. Dựa trên các thông số đo đạc tại thời điểm hiện tại ($t$), mô hình sẽ dự báo chỉ số ô nhiễm tại bước thời gian tiếp theo ($t+1$). Kết quả dự báo là cơ sở hỗ trợ ban quản lý đưa ra quyết định can thiệp sớm, tránh rủi ro xả thải vượt quy chuẩn ra môi trường.
- **Đặc trưng đầu vào (Features - X)**: Tập dữ liệu bao gồm các chỉ số vật lý, hóa học và động lực học đo lường tự động: 
  - `pH`, `Temp` (Nhiệt độ).
  - `TSS` (Tổng lượng chất rắn lơ lửng).
  - `NH4+` (Nồng độ Amoni).
  - `Color` (Độ màu).
  - `Flow1`, `Flow2` (Lưu lượng dòng chảy - yếu tố biến động ảnh hưởng trực tiếp đến nồng độ ô nhiễm).
- **Biến mục tiêu dự báo (Target - y)**: `COD` (Nhu cầu Oxy hóa học) - Thông số định lượng cốt lõi trong việc đánh giá mức độ ô nhiễm hữu cơ của nước thải công nghiệp.


## Bước 1: Thiết lập cấu trúc & Môi trường
- **Mục tiêu**: Xây dựng nền tảng dự án chuẩn hóa.
- **Công việc**: 
  - Tạo cấu trúc thư mục dạng module (data, models, src, ...).
  - Cài đặt các thư viện cần thiết (Pandas, Scikit-learn, XGBoost, SHAP, Streamlit).

## Bước 2: Tiền xử lý dữ liệu (Data Preprocessing)
- **Mục tiêu**: Làm sạch và chuẩn bị dữ liệu đầu vào.
- **Công việc**:
  - Gộp các file log `.txt` từ nhiều trạm quan trắc thành một dataset duy nhất.
  - **Điền dữ liệu thiếu**: Sử dụng phương pháp Trung vị (Median Imputation).
  - **Lọc nhiễu (Outliers)**: Dùng phương pháp Khoảng tứ phân vị (IQR).
  - **Chuẩn hóa**: Đưa các đặc trưng về cùng thang đo (Min-Max Scaling).

## Bước 3: Huấn luyện mô hình (Modeling)
- **Mục tiêu**: Tìm ra mô hình dự báo COD tốt nhất.
- **Công việc**:
  - Huấn luyện 6 mô hình cơ sở: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, CatBoost.
  - Xây dựng **Stacking Regressor**: Dùng Linear Regression làm meta-learner để kết hợp ưu điểm của 6 mô hình trên.
  - Áp dụng Cross-Validation (5-fold) để tránh quá khớp.
  - So sánh các chỉ số: $R^2$, Adjusted $R^2$, RMSE, MAE.

## Bước 4: Trí tuệ nhân tạo có thể giải thích (SHAP)
- **Mục tiêu**: Tăng tính minh bạch, giải thích được lý do đằng sau dự đoán của mô hình học máy (đáp ứng tiêu chí điểm sáng tạo).
- **Công việc**:
  - Tính toán giá trị SHAP (TreeExplainer).
  - Trực quan hóa mức độ đóng góp của từng đặc trưng (Feature Importance).
  - Phân tích tương tác cục bộ và toàn cục (Summary Plot, Force Plot).

## Bước 5: Triển khai Giao diện Web (Streamlit)
- **Mục tiêu**: Đóng gói dự án thành một sản phẩm trực quan.
- **Công việc**:
  - Tạo web app bằng Streamlit.
  - Cho phép người dùng nhập các thông số đầu vào.
  - Hiển thị kết quả dự báo COD và biểu đồ giải thích SHAP theo thời gian thực.
