"""
Cấu hình chung cho pipeline v2.
Mọi script trong src/ và run_all.py đều đọc hằng số từ đây để đảm bảo nhất quán.
"""
import os

# ---------------- Đường dẫn ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "v2")   # không ghi đè file cũ
MODELS_DIR = os.path.join(BASE_DIR, "models", "v2")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "v2")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

for _d in (PROCESSED_DIR, MODELS_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- Dữ liệu ----------------
FEATURES = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
TARGET = 'COD'
UNITS = {'COD': 'mg/L', 'TSS': 'mg/L', 'pH': '-', 'Temp': '°C', 'NH4+': 'mg/L',
         'Color': 'Pt-Co', 'Flow2': 'm³/h', 'Flow1': 'm³/h'}
STEP_MIN = 5                 # chu kỳ quan trắc (phút)
TRAIN_FRAC = 0.8             # 80% đầu theo thời gian là train
VALID_STATUS = {'00'}        # mã trạng thái hợp lệ (TT 10/2021/TT-BTNMT: 00 = đang đo bình thường)
IQR_K = 1.5

# Có clip IQR cho COD hay không. COD vừa là đặc trưng COD(t) vừa là mục tiêu COD(t+1);
# clip COD đồng nghĩa với việc sửa "đáp án" -> mặc định KHÔNG clip COD.
# Đặt True để tái hiện cách làm cũ (dùng cho phân tích độ nhạy).
CLIP_COD = False

# ---------------- Huấn luyện ----------------
SEED = 42
SEEDS = [42, 43, 44, 45, 46]
N_SPLITS_TS = 5              # TimeSeriesSplit cho stacking
N_TIMING_REPEATS = 7         # số lần lặp đo suy luận (lấy median)

# Tầm dự báo (số bước 5 phút): 5 phút, 30 phút, 1 giờ, 2 giờ
HORIZONS = [1, 6, 12, 24]

# ---------------- SHAP ----------------
SHAP_BACKGROUND_N = 200      # mẫu nền lấy từ train
SHAP_TEST_N = 500            # số mẫu test để tính SHAP (nếu ít hơn toàn bộ test -> ghi rõ là lấy mẫu)
