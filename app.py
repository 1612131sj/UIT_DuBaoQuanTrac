import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- 1. SETUP PAGE ---
# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hệ thống Dự báo Ô nhiễm KCN", layout="wide", page_icon="💧")

# Đảm bảo hiển thị font rõ nét
plt.rcParams.update({'font.size': 10})

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    """Hàm tải mô hình (dùng cache để web không bị chậm khi tải lại)"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Dùng StackingEnsemble để dự đoán cho chính xác nhất
    model = joblib.load(os.path.join(BASE_DIR, "models", "StackingEnsemble.pkl"))
    # Dùng ExtraTrees để SHAP giải thích cho nhanh và tương thích tốt
    shap_model = joblib.load(os.path.join(BASE_DIR, "models", "ExtraTrees.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "minmax_scaler.pkl"))
    return model, shap_model, scaler

try:
    model, shap_model, scaler = load_models()
    features = ['COD', 'TSS', 'pH', 'Temp', 'NH4+', 'Color', 'Flow2', 'Flow1']
except Exception as e:
    st.error(f"Lỗi tải mô hình. Vui lòng đảm bảo bạn đã chạy Bước 3 (train_pipeline.py) thành công. Lỗi: {e}")
    st.stop()

# --- 3. UI: GIAO DIỆN CHÍNH ---
st.title("💧 Hệ thống Cảnh báo Sớm Ô nhiễm Nước thải")
st.markdown("""
Ứng dụng **Trí tuệ nhân tạo (AI)** để dự báo nồng độ COD ở thời điểm tương lai ($t+1$) dựa trên dữ liệu trạm quan trắc hiện tại ($t$).
""")

st.subheader("📊 1. Điều chỉnh Thông số Cảm biến Hiện tại")

# Tạo 4 cột trên màn hình chính
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        cod = st.slider("COD (mg/L)", 0.0, 150.0, 15.0, 0.5)
    with st.container(border=True):
        nh4 = st.slider("NH4+ (mg/L)", 0.0, 50.0, 1.2, 0.1)

with col2:
    with st.container(border=True):
        tss = st.slider("TSS (mg/L)", 0.0, 100.0, 10.0, 0.5)
    with st.container(border=True):
        color = st.slider("Color (Pt-Co)", 0.0, 100.0, 15.0, 0.5)

with col3:
    with st.container(border=True):
        ph = st.slider("pH", 0.0, 14.0, 7.5, 0.1)
    with st.container(border=True):
        flow1 = st.slider("Flow1 (m3/h)", 0.0, 200.0, 73.0, 1.0)

with col4:
    with st.container(border=True):
        temp = st.slider("Nhiệt độ (oC)", 20.0, 45.0, 32.0, 0.1)
    with st.container(border=True):
        flow2 = st.slider("Flow2 (m3/h)", 0.0, 200.0, 81.0, 1.0)

# Gom dữ liệu thành DataFrame
input_df = pd.DataFrame({
    'COD': [cod], 'TSS': [tss], 'pH': [ph], 'Temp': [temp],
    'NH4+': [nh4], 'Color': [color], 'Flow2': [flow2], 'Flow1': [flow1]
})

st.divider()

# --- 5. PREDICT & EXPLAIN ---
st.subheader("🎯 2. Kết quả Dự báo (t+1)")

if st.button("🚀 Thực hiện Dự báo", type="primary"):
    with st.spinner('AI đang tính toán...'):
        # Chuẩn hóa dữ liệu đầu vào
        input_scaled = scaler.transform(input_df)
        
        # Dự đoán
        prediction = model.predict(input_scaled)[0]
        
        # Khung hiển thị kết quả trực quan
        st.markdown("### Nồng độ COD dự báo:")
        
        # Phân loại cảnh báo
        if prediction > 75:
            st.error(f"## {prediction:.2f} mg/L\n⚠️ CẢNH BÁO ĐỎ: Vượt tiêu chuẩn A (Nguy cơ ô nhiễm cao!)")
        elif prediction > 30:
            st.warning(f"## {prediction:.2f} mg/L\n⚡ MỨC TRUNG BÌNH: Nước bắt đầu có dấu hiệu ô nhiễm.")
        else:
            st.success(f"## {prediction:.2f} mg/L\n✅ AN TOÀN: Đạt tiêu chuẩn xả thải.")
            
        st.divider()
        st.subheader("🧠 3. Trí tuệ nhân tạo Giải thích")
        st.markdown("Biểu đồ SHAP thể hiện lý do: Yếu tố nào đã **đẩy COD tăng lên** (Màu Đỏ) và yếu tố nào giúp **kéo COD giảm xuống** (Màu Xanh).")
        
        # Tính toán SHAP
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(input_scaled)
        
        # Vẽ biểu đồ Local Impact bằng Matplotlib
        # Đặt facecolor='white' để chữ không bị chìm vào nền đen của giao diện Dark Mode
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='black', layout='constrained')
        ax.set_facecolor('black')
        
        sv = shap_values[0] # Lấy mảng SHAP của người dùng nhập
        
        shap_data = pd.DataFrame({'Feature': features, 'SHAP Value': sv})
        shap_data['Positive'] = shap_data['SHAP Value'] > 0
        shap_data = shap_data.sort_values(by='SHAP Value', key=abs)
        
        colors = shap_data['Positive'].map({True: 'crimson', False: 'cornflowerblue'})
        bars = ax.barh(shap_data['Feature'], shap_data['SHAP Value'], color=colors)
        
        # Thêm con số (giá trị SHAP) lên từng thanh
        ax.bar_label(bars, fmt=' %+.2f', padding=3, fontweight='bold')
        
        # Căn chỉnh lại biểu đồ cho đẹp
        ax.set_xlabel("Mức độ tác động đến kết quả dự báo (SHAP Value)")
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Đảm bảo chữ không bị cắt xén
        fig.tight_layout()
        
        st.pyplot(fig)
else:
    st.info("👈 Vui lòng bấm nút 'Thực hiện Dự báo' để xem kết quả.")

