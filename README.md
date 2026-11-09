# 💧 Dự báo COD nước thải KCN bằng Stacked Ensemble & SHAP

Đồ án môn học: **Học máy tiên tiến và ứng dụng**  
Nghiên cứu ứng dụng phương pháp Học máy kết hợp (*Stacked Ensemble Regression*) và Trí tuệ nhân tạo có thể giải thích (*SHAP Explainable AI*) để dự báo ngắn hạn nồng độ COD trong nước thải Khu công nghiệp.

---

## 🎯 Tổng quan Dự án
- **Bài toán**: Dùng dữ liệu quan trắc tự động tại thời điểm $t$ (trạm LA_KCN_ANHHON, chu kỳ 5 phút, 01–23/01/2026) để dự báo nồng độ COD tại $t+1$ (5 phút sau); thử nghiệm thêm các tầm 30 phút, 1 giờ, 2 giờ.
- **Đặc trưng đầu vào**: `COD`, `TSS`, `pH`, `Temp`, `NH4+`, `Color`, `Flow1`, `Flow2` tại thời điểm $t$ (tầm dự báo dài hơn: thêm trễ và trung bình/độ lệch chuẩn trượt của COD).
- **Biến mục tiêu**: `COD` tại $t+h$ (mg/L).

## 🧠 Phương pháp
- **Tiền xử lý không rò rỉ**: chia train/test 80/20 theo thời gian trước; điền thiếu (forward-fill) và ngưỡng clip IQR chỉ học trên train; không tạo cặp $(t, t+h)$ qua khoảng trống thời gian.
- **Baseline**: Persistence ($\hat{y}_{t+1} = COD_t$) và Linear Regression.
- **Mô hình cơ sở**: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, CatBoost.
- **Stacking**: meta-learner Linear Regression học trên dự báo out-of-fold của `TimeSeriesSplit(5)` (cài đặt trong `src/ensemble_models.py`).
- **Đánh giá**: R², Adj R², RMSE, MAE, MAPE, Skill so với Persistence; 5 seed; kiểm định Diebold–Mariano và Wilcoxon; đo thời gian huấn luyện/suy luận.
- **Giải thích**: SHAP exact cho chính mô hình Stacking (thang mg/L), đối chiếu TreeSHAP của XGBoost/CatBoost.

Kết quả chi tiết và nhận xét: `outputs/v2/RESULTS.md`.

---

## 📂 Cấu trúc Thư mục

```text
PROJECT/
├── data/
│   ├── raw/                     # Dữ liệu gốc (.txt) trích xuất từ trạm quan trắc
│   └── processed/v2/            # Dữ liệu gộp (có cột trạng thái) và dữ liệu sau tiền xử lý v2
├── models/v2/                   # Mô hình (.pkl), khoảng min–max của train, mẫu nền SHAP
├── outputs/v2/                  # Kết quả v2: tables/, figures/, environment.txt, RESULTS.md
├── src/
│   ├── config.py                # Hằng số, đường dẫn, seed, horizon
│   ├── data_preprocessing.py    # Việc 1–2: audit dữ liệu, tiền xử lý fit trên train, tạo cặp (t, t+h)
│   ├── ensemble_models.py       # Persistence, LinearRegression, 6 mô hình cây, Stacking theo TimeSeriesSplit
│   ├── evaluation.py            # Chỉ số đánh giá, Diebold–Mariano, Wilcoxon
│   ├── train_pipeline.py        # Việc 3–4: huấn luyện 5 seed, kiểm định, đo thời gian
│   ├── horizon_experiment.py    # Việc 5: tầm dự báo 5/30/60/120 phút
│   ├── residual_plots.py        # Hình phần dư + kiểm định phần dư
│   ├── generate_paper_plots.py  # Các hình còn lại cho báo cáo
│   ├── explainability.py        # SHAP cho chính mô hình Stacking (+ đối chiếu TreeSHAP)
│   ├── make_results.py          # Sinh outputs/v2/RESULTS.md từ các bảng CSV
│   └── environment_info.py      # Ghi phiên bản thư viện, CPU, RAM, OS
├── run_all.py                   # Chạy toàn bộ pipeline v2
├── app.py                       # Giao diện Web Streamlit
├── requirements.txt
```

Kết quả phiên bản cũ (thư mục `outputs/` gốc) được giữ nguyên để đối chiếu.

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

### 2. Chạy toàn bộ pipeline
```bash
python run_all.py
```
Lệnh này lần lượt: ghi thông tin môi trường → kiểm tra & tiền xử lý dữ liệu → huấn luyện và kiểm định → thí nghiệm
tầm dự báo → vẽ hình → tính SHAP → sinh `outputs/v2/RESULTS.md`. Có thể bỏ qua các bước lâu bằng
`python run_all.py --skip-horizon --skip-shap`. Mỗi script trong `src/` cũng chạy độc lập được (`python src/<tên>.py`).

### 3. Mở giao diện Web (Streamlit)
```bash
streamlit run app.py
```
Ngưỡng cảnh báo COD lấy theo QCVN 40:2025/BTNMT (theo lưu lượng xả F và cột A/B/C), cấu hình trong dict `QCVN_40_2025` của `app.py`.

---

## 📌 Khuyến nghị cho Dữ liệu Mới
Nếu cần thay đổi một bộ dữ liệu quan trắc mới (từ một khu công nghiệp khác), các bước thực hiện như sau:
1. Xóa các file trong `data/processed/v2/`, `models/v2/` và `outputs/v2/`.
2. Đặt dữ liệu mới vào `data/raw/` và chạy lại `python run_all.py` để hệ thống học lại từ đầu quy luật của dòng nước mới.
