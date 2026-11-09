"""
Ứng dụng Streamlit dự báo COD(t+1).
Chạy: streamlit run app.py   (cần chạy `python run_all.py` trước để có models/v2/)

- Thanh trượt giới hạn trong [min, max] của tập train; chế độ nhập tự do có cảnh báo ngoài khoảng train.
- Ngưỡng cảnh báo COD theo QCVN 40:2025/BTNMT đặt trong dict cấu hình, chọn nhóm lưu lượng xả F và cột A/B/C.
- SHAP waterfall tính cho chính mô hình đang dự báo (exact Shapley, nền 200 mẫu train, thang mg/L).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(BASE_DIR, "src"))   # để joblib tìm được lớp trong src/ensemble_models.py
MODELS_DIR = os.path.join(BASE_DIR, "models", "v2")
TAB_DIR = os.path.join(BASE_DIR, "outputs", "v2", "tables")
FEATURES = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
LABELS = {'COD': 'COD (mg/L)', 'TSS': 'TSS (mg/L)', 'pH': 'pH', 'Temp': 'Nhiệt độ (°C)',
          'NH4+': 'NH4+ (mg/L)', 'Color': 'Độ màu (Pt-Co)', 'Flow2': 'Flow2 (m³/h)', 'Flow1': 'Flow1 (m³/h)'}

# ------------------------------------------------------------------
# CẤU HÌNH NGƯỠNG COD THEO QCVN 40:2025/BTNMT
# Nguồn: QCVN 40:2025/BTNMT – Quy chuẩn kỹ thuật quốc gia về nước thải công nghiệp
# (tóm tắt: https://luatvietnam.vn/doanh-nghiep/gioi-han-cho-phep-o-nhiem-nuoc-thai-cong-nghiep-btnmt-561-101346-article.html).
# Giới hạn COD (mg/L) phụ thuộc lưu lượng xả thải F (m³/ngày) và cột A/B/C của nguồn tiếp nhận.
# Giá trị None nghĩa là chưa cấu hình: ứng dụng chỉ hiển thị dự báo, không phân loại cảnh báo.
# ------------------------------------------------------------------
QCVN_40_2025 = {
    'F ≤ 2.000 m³/ngày': {'A': 65, 'B': 90, 'C': 130},
    'F > 2.000 m³/ngày': {'A': 60, 'B': 70, 'C': 90},
}
WARNING_RATIO = 0.8   # mức "cảnh giác" khi dự báo ≥ 80% ngưỡng

st.set_page_config(page_title="Dự báo COD – KCN", layout="wide", page_icon="💧")


@st.cache_resource
def load_artifacts():
    models = {}
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith('.pkl'):
            models[f[:-4]] = joblib.load(os.path.join(MODELS_DIR, f))
    with open(os.path.join(MODELS_DIR, "train_ranges.json"), encoding='utf-8') as f:
        ranges = json.load(f)
    bg = np.load(os.path.join(MODELS_DIR, "shap_background.npy"))
    metrics = pd.read_csv(os.path.join(TAB_DIR, "model_metrics.csv")).set_index('Model')
    return models, ranges, bg, metrics


@st.cache_resource
def get_explainer(model_name):
    model = MODELS[model_name]
    return shap.Explainer(model.predict, shap.maskers.Independent(BG, max_samples=len(BG)),
                          algorithm='exact', feature_names=FEATURES)


try:
    MODELS, RANGES, BG, METRICS = load_artifacts()
except Exception as e:
    st.error(f"Không tải được mô hình trong models/v2/. Hãy chạy `python run_all.py` trước. Lỗi: {e}")
    st.stop()

# ------------------------------- Thanh bên -------------------------------
st.sidebar.header("⚙️ Cấu hình")
names = [m for m in METRICS.index if m in MODELS]
learned = [m for m in names if m != 'Persistence']
default = METRICS.loc[learned, 'RMSE_mean'].idxmin()
model_name = st.sidebar.selectbox(
    "Mô hình dự báo", names, index=names.index(default),
    format_func=lambda m: f"{m} (RMSE test = {METRICS.loc[m, 'RMSE_mean']:.3f} mg/L)")
flow_groups = list(QCVN_40_2025)
flow_group = st.sidebar.radio("Lưu lượng xả thải F (QCVN 40:2025/BTNMT)", flow_groups, index=1)
st.sidebar.caption(f"Tham khảo: trung vị Flow1 ≈ {RANGES['Flow1'][2] * 24:,.0f} m³/ngày, "
                   f"Flow2 ≈ {RANGES['Flow2'][2] * 24:,.0f} m³/ngày (m³/h × 24, dữ liệu train). "
                   "Mặc định chọn nhóm F > 2.000 m³/ngày (ngưỡng chặt hơn).".replace(',', '.'))
col = st.sidebar.radio("Cột quy chuẩn", ['A', 'B', 'C'], horizontal=True,
                       format_func=lambda c: f"Cột {c} (≤ {QCVN_40_2025[flow_group][c]} mg/L)")
free_input = st.sidebar.checkbox("Nhập số tự do (cho phép ngoài khoảng train)", value=False)
st.sidebar.caption("Mặc định chọn mô hình học máy có RMSE test thấp nhất. "
                   "Persistence (ŷ(t+1) = COD(t)) là baseline để đối chiếu.")

# ------------------------------- Nhập liệu -------------------------------
st.title("💧 Dự báo COD chu kỳ kế tiếp (t + 5 phút)")
st.markdown("Dự báo nồng độ COD ở chu kỳ quan trắc kế tiếp (5 phút) từ 8 thông số tại thời điểm hiện tại "
            "của trạm LA_KCN_ANHHON.")
st.subheader("1. Thông số cảm biến hiện tại (t)")

values = {}
cols = st.columns(4)
for i, f in enumerate(FEATURES):
    lo, hi, med = RANGES[f]
    with cols[i % 4]:
        with st.container(border=True):
            if free_input:
                values[f] = st.number_input(LABELS[f], value=float(round(med, 2)), step=0.1, format="%.2f")
            else:
                values[f] = st.slider(LABELS[f], float(round(lo, 2)), float(round(hi, 2)),
                                      float(round(med, 2)), 0.01)
            st.caption(f"Khoảng train: {lo:.2f} – {hi:.2f}")

x = np.array([[values[f] for f in FEATURES]], dtype=float)
outside = [f for f in FEATURES if not (RANGES[f][0] <= values[f] <= RANGES[f][1])]
if outside:
    st.warning("⚠️ Các thông số nằm NGOÀI khoảng dữ liệu huấn luyện: "
               + ", ".join(f"{LABELS[f]} = {values[f]:.2f} (train: {RANGES[f][0]:.2f}–{RANGES[f][1]:.2f})"
                           for f in outside)
               + ". Mô hình dạng cây không ngoại suy được – kết quả kém tin cậy.")

st.divider()
st.subheader(f"2. Kết quả dự báo COD(t+1) – mô hình {model_name}")

if st.button("🚀 Thực hiện dự báo", type="primary"):
    model = MODELS[model_name]
    pred = float(model.predict(x)[0])
    thr = QCVN_40_2025[flow_group][col]
    label = f"cột {col}, {flow_group}"
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("COD dự báo (mg/L)", f"{pred:.2f}", delta=f"{pred - values['COD']:+.2f} so với COD(t)",
                  delta_color="inverse")
    with c2:
        if thr is None:
            st.info(f"Chưa cấu hình ngưỡng COD cho {label} trong `QCVN_40_2025` (app.py) – "
                    "không phân loại cảnh báo.")
        elif pred > thr:
            st.error(f"🔴 VƯỢT ngưỡng QCVN 40:2025 ({label}): {pred:.2f} > {thr} mg/L")
        elif pred >= WARNING_RATIO * thr:
            st.warning(f"🟠 Gần ngưỡng QCVN 40:2025 ({label}): {pred:.2f} mg/L ≥ "
                       f"{WARNING_RATIO:.0%} × {thr} mg/L")
        else:
            st.success(f"🟢 Dưới ngưỡng QCVN 40:2025 ({label}): {pred:.2f} ≤ {thr} mg/L")

    st.subheader("3. Giải thích dự báo (SHAP waterfall)")
    st.caption(f"Giá trị Shapley chính xác (thuật toán exact, 2⁸ tổ hợp) cho chính mô hình {model_name}, "
               f"nền = {len(BG)} mẫu ngẫu nhiên từ train, đơn vị mg/L. "
               "Thanh đỏ đẩy dự báo lên, thanh xanh kéo xuống so với giá trị nền E[f(X)].")
    with st.spinner("Đang tính SHAP..."):
        sv = get_explainer(model_name)(x)
    ex = shap.Explanation(values=sv.values[0], base_values=float(np.atleast_1d(sv.base_values)[0]),
                          data=x[0], feature_names=[LABELS[f] for f in FEATURES])
    plt.close('all')
    shap.plots.waterfall(ex, max_display=8, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 4.5)
    fig.patch.set_facecolor('white')
    st.pyplot(fig, clear_figure=True)
    st.dataframe(pd.DataFrame({'Thông số': [LABELS[f] for f in FEATURES], 'Giá trị (t)': x[0],
                               'SHAP (mg/L)': sv.values[0]}).sort_values('SHAP (mg/L)', key=abs,
                                                                         ascending=False),
                 hide_index=True, use_container_width=True)
else:
    st.info("Bấm **Thực hiện dự báo** để xem kết quả.")
