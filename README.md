# 💧 Dự báo Cảnh báo sớm Ô nhiễm Nước thải KCN (Stacked Ensemble & SHAP)

Đồ án môn học: **Học máy tiên tiến và ứng dụng**  
Nghiên cứu ứng dụng phương pháp Học máy kết hợp (*Stacked Ensemble Regression*) và Trí tuệ nhân tạo có thể giải thích (*SHAP Explainable AI*) để dự báo sớm chất lượng nước thải tại các Khu công nghiệp.

---

## 🎯 Tổng quan Dự án
- **Bài toán**: Khai thác dữ liệu chuỗi thời gian từ hệ thống quan trắc tự động tại thời điểm $t$ để dự báo nồng độ ô nhiễm hữu cơ (COD) tại thời điểm $t+1$ (ví dụ: 5 phút sau).
- **Ý nghĩa**: Cung cấp cơ sở để ban quản lý đưa ra quyết định can thiệp sớm (ví dụ: đóng van xả, tăng cường xử lý) nhằm ngăn chặn rủi ro xả thải vi phạm quy chuẩn ra môi trường.
- **Đặc trưng đầu vào (Features)**: `pH`, `Temp`, `TSS`, `NH4+`, `Color`, `Flow1`, `Flow2`.
- **Biến mục tiêu (Target)**: `COD` (Nhu cầu Oxy hóa học).

## 🧠 Kiến trúc Học máy
Dự án sử dụng **Stacking Regressor** (Mô hình kết hợp) để tổng hợp sức mạnh từ 6 mô hình cơ sở mạnh mẽ nhất hiện nay:
1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. AdaBoost
5. XGBoost
6. CatBoost

*(Meta-learner sử dụng Linear Regression).*

Để tránh tình trạng "hộp đen" (Black-box), dự án tích hợp thư viện **SHAP** (`shap.TreeExplainer`) nhằm trực quan hóa mức độ đóng góp của từng thông số cảm biến vào quyết định tăng/giảm dự báo COD của AI.

---

## 📂 Cấu trúc Thư mục

```text
PROJECT/
├── data/
│   ├── raw/                   # Dữ liệu gốc (.txt) trích xuất từ trạm quan trắc
│   └── processed/             # Dữ liệu đã gộp và làm sạch (.csv)
├── models/                    # Lưu trữ các mô hình AI (.pkl) và Scaler
├── outputs/                   # Các biểu đồ SHAP và file đánh giá kết quả (.csv)
├── src/
│   ├── data_preprocessing.py  # Xử lý nhiễu (IQR), điền thiếu (Median), Pivot dữ liệu
│   ├── train_pipeline.py      # Huấn luyện 6 mô hình + Stacking, chuẩn hóa Min-Max
│   └── explainability.py      # Trực quan hóa suy luận của AI bằng SHAP
├── app.py                     # Giao diện Web tương tác bằng Streamlit
├── requirements.txt           # Danh sách thư viện cần thiết
└── project_roadmap.md         # Tài liệu dàn ý báo cáo/thuyết trình
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Dự án

### 1. Thiết lập môi trường
Khuyến nghị sử dụng môi trường ảo (`venv`) để tránh xung đột thư viện:

```bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường (trên Windows)
.\venv\Scripts\activate

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Quy trình chạy mã nguồn
Thực hiện chạy lần lượt các script theo thứ tự sau (đứng ở thư mục gốc của dự án):

**Bước 1: Tiền xử lý dữ liệu**
Chuyển đổi dữ liệu thô dạng file `.txt` thành bảng dữ liệu `.csv` đã được làm sạch nhiễu.
```bash
python src/data_preprocessing.py
```

**Bước 2: Huấn luyện mô hình**
Tiến hành chia tập Train/Test, chuẩn hóa Min-Max, và huấn luyện 7 thuật toán Học máy.
```bash
python src/train_pipeline.py
```

**Bước 3: Giải thích mô hình (SHAP)**
Vẽ biểu đồ Bar Plot và Summary Plot để phân tích tầm quan trọng của các thông số.
```bash
python src/explainability.py
```

**Bước 4: Mở Giao diện Web (Streamlit)**
Chạy Dashboard tương tác để kiểm thử hệ thống cảnh báo sớm.
```bash
streamlit run app.py
```

---

## 📌 Khuyến nghị cho Dữ liệu Mới
Nếu bạn thay đổi một bộ dữ liệu quan trắc mới (từ một khu công nghiệp khác), hãy đảm bảo:
1. Xóa các file cũ trong thư mục `data/processed/`, `models/` và `outputs/`.
2. Đặt dữ liệu mới vào `data/raw/` và chạy lại từ **Bước 1** đến **Bước 4** để hệ thống học lại từ đầu quy luật của dòng nước mới.
