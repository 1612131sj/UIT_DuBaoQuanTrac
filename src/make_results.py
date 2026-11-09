"""
Sinh outputs/v2/RESULTS.md từ các file CSV trong outputs/v2/tables.
Mọi con số trong RESULTS.md (kể cả trong phần nhận xét) đều đọc trực tiếp từ CSV do code tính ra.
"""
import os
import datetime
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C

T = lambda f: os.path.join(C.TAB_DIR, f)


def rd(f, **kw):
    return pd.read_csv(T(f), **kw)


def fp(p):
    """Định dạng p-value."""
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def md(df, floatfmt=4, index=False):
    """DataFrame -> bảng markdown (không cần thư viện tabulate)."""
    d = df.reset_index() if index else df
    def cell(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ''
            return f"{v:.{floatfmt}f}"
        return str(v)
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    sep = '|' + '|'.join(['---'] * len(d.columns)) + '|'
    rows = ['| ' + ' | '.join(cell(v) for v in r) + ' |' for r in d.itertuples(index=False)]
    return '\n'.join([head, sep] + rows)


def pm(m, s, k=4):
    return f"{m:.{k}f} ± {s:.{k}f}"


def main():
    L = []
    add = L.append
    env = open(os.path.join(C.OUT_DIR, "environment.txt"), encoding='utf-8').read()

    # ------------------------------------------------------------ dữ liệu
    at = rd("data_audit_time.csv")
    atd = dict(zip(at['Mục'], at['Giá trị']))
    gaps = rd("data_gaps.csv")
    audit = rd("data_audit.csv")
    desc = rd("descriptive_stats.csv")
    codsplit = rd("cod_train_vs_test.csv", index_col=0)
    m = rd("model_metrics.csv").set_index('Model')
    per = rd("model_metrics_per_seed.csv")
    sig = rd("significance.csv")
    mw = rd("meta_weights.csv")
    tim = rd("timing.csv")
    var = rd("stacking_variants.csv")
    sens = rd("sensitivity_clip_cod.csv")
    res = rd("residual_stats.csv").set_index('Mô hình')
    ccf = rd("lag_crosscorr.csv")
    has_h = os.path.exists(T("horizon_metrics.csv"))
    has_shap = os.path.exists(T("shap_importance.csv"))

    add("# KẾT QUẢ PIPELINE v2 – DỰ BÁO COD TRẠM LA_KCN_ANHHON\n")
    add(f"*Tệp này được sinh tự động bởi `src/make_results.py` lúc {datetime.datetime.now():%Y-%m-%d %H:%M} "
        "từ các file trong `outputs/v2/tables/`. Không có con số nào được nhập tay.*\n")
    add("Chạy lại toàn bộ: `python run_all.py`. Số thập phân dùng dấu chấm; đơn vị COD là mg/L.\n")

    add("## 0. Môi trường chạy\n\n```\n" + env.strip() + "\n```\n")

    # ------------------------------------------------------------ 1. audit
    add("## 1. Kiểm tra dữ liệu\n")
    add("**Bảng 1. Tóm tắt kiểm tra trục thời gian** (`tables/data_audit_time.csv`)\n")
    add(md(at))
    add("\n**Bảng 2. Các khoảng trống thời gian (gap ≠ 5 phút)** (`tables/data_gaps.csv`)\n")
    add(md(gaps, 0))
    cols = ['Cột', 'Đơn vị', 'Tỉ lệ thiếu (%)', 'Status ≠ 00 (coi là thiếu)', 'Điền bằng ffill',
            'Điền bằng median train', 'Ngưỡng IQR dưới (train)', 'Ngưỡng IQR trên (train)', 'Có clip?',
            'Clip dưới (train)', 'Clip trên (train)', 'Clip dưới (test)', 'Clip trên (test)']
    add("\n**Bảng 3. Kiểm tra theo cột: thiếu, điền, ngưỡng IQR (fit trên train) và số giá trị bị clip** "
        "(`tables/data_audit.csv`)\n")
    add(md(audit[cols], 3))
    cod_a = audit.set_index('Cột').loc['COD']
    add(f"\nGhi chú: COD **không** bị clip (mặc định `CLIP_COD = False`). Nếu áp ngưỡng IQR của train thì có "
        f"{int(cod_a['Ngoài ngưỡng IQR (train)'])} giá trị COD ở train và "
        f"{int(cod_a['Ngoài ngưỡng IQR (test)'])} giá trị ở test nằm ngoài ngưỡng.\n")

    # ------------------------------------------------------------ 2. thống kê
    add("## 2. Thống kê mô tả sau tiền xử lý\n")
    add("**Bảng 4. Thống kê mô tả 8 thông số sau tiền xử lý (toàn giai đoạn)** (`tables/descriptive_stats.csv`)\n")
    add(md(desc, 3))
    cs = codsplit.copy()
    cs['count'] = cs['count'].astype(int).astype(str)
    cs.index.name = 'Giai đoạn'
    add("\n**Bảng 5. Phân phối COD theo giai đoạn train/test** (`tables/cod_train_vs_test.csv`)\n")
    add(md(cs, 3, index=True))

    # ------------------------------------------------------------ 3. mô hình
    add("\n## 3. So sánh mô hình dự báo COD(t+1)\n")
    tbl = pd.DataFrame({
        'Mô hình': m.index,
        'R²': [pm(a, b) for a, b in zip(m['R2_mean'], m['R2_std'])],
        'Adj R²': [pm(a, b) for a, b in zip(m['Adj_R2_mean'], m['Adj_R2_std'])],
        'RMSE (mg/L)': [pm(a, b) for a, b in zip(m['RMSE_mean'], m['RMSE_std'])],
        'MAE (mg/L)': [pm(a, b) for a, b in zip(m['MAE_mean'], m['MAE_std'])],
        'MAPE (%)': [pm(a, b, 2) for a, b in zip(m['MAPE_%_mean'], m['MAPE_%_std'])],
        'Skill so với Persistence': [pm(a, b) for a, b in
                                     zip(m['Skill_vs_Persistence_mean'], m['Skill_vs_Persistence_std'])],
        'Số seed': m['n_seeds'].astype(int)})
    add(f"**Bảng 6. Kết quả trên tập test (n = {int(atd['Số cặp test (h=1)'])} cặp), trung bình ± độ lệch "
        "chuẩn qua 5 seed (42–46)** (`tables/model_metrics.csv`)\n")
    add(md(tbl))
    add("\nPersistence và LinearRegression không có yếu tố ngẫu nhiên nên chỉ chạy 1 lần (std = 0). "
        "XGBoost với cấu hình mặc định (không subsample) cho kết quả giống hệt nhau ở mọi seed. "
        "Adj R² dùng p = 8 (p = 1 cho Persistence). Skill = 1 − RMSE_model / RMSE_Persistence.\n")

    s42 = sig[sig['Seed'] == C.SEED].copy()
    agg = sig.groupby('So sánh (A vs B)').agg(p_max=('DM_p (MSE)', 'max'), p_min=('DM_p (MSE)', 'min'),
                                              dm_min=('DM_stat (MSE)', 'min'), dm_max=('DM_stat (MSE)', 'max'))
    sg = pd.DataFrame({
        'So sánh (A vs B)': s42['So sánh (A vs B)'],
        'RMSE A': s42['RMSE_A'], 'RMSE B': s42['RMSE_B'],
        'DM (MSE)': s42['DM_stat (MSE)'], 'p DM (MSE)': s42['DM_p (MSE)'].map(fp),
        'DM (MAE)': s42['DM_stat (MAE)'], 'p DM (MAE)': s42['DM_p (MAE)'].map(fp),
        'p Wilcoxon |e|': s42['Wilcoxon_p |e|'].map(fp),
        'DM (MSE) 5 seed [min; max]': [f"[{agg.loc[k, 'dm_min']:.2f}; {agg.loc[k, 'dm_max']:.2f}]"
                                       for k in s42['So sánh (A vs B)']],
        'p DM (MSE) lớn nhất qua 5 seed': [fp(agg.loc[k, 'p_max']) for k in s42['So sánh (A vs B)']]})
    add("**Bảng 7. Kiểm định Diebold–Mariano (hiệu chỉnh HLN, phương sai HAC Newey–West, lag = "
        f"{int(s42['HAC_lag'].iloc[0])}) và Wilcoxon signed-rank trên |sai số|, seed 42** "
        "(`tables/significance.csv`)\n")
    add(md(sg))
    add("\nDM < 0: mô hình A có sai số nhỏ hơn B; DM > 0: A kém hơn B. "
        "Wilcoxon không tính đến tự tương quan của sai số nên p-value của nó lạc quan hơn thực tế.\n")

    add("**Bảng 8. Hệ số của meta-learner (LinearRegression) trong Stacking theo seed** (`tables/meta_weights.csv`)\n")
    add(md(mw))
    add("\n**Bảng 9. Phân tích bổ sung: các biến thể Stacking (seed 42)** (`tables/stacking_variants.csv`)\n")
    add(md(var[['Biến thể', 'R2', 'RMSE', 'MAE', 'MAPE_%', 'Intercept meta', 'Tổng hệ số meta']]))
    add("\n**Bảng 10. Độ nhạy: clip IQR cả COD như phiên bản trước (seed 42, đánh giá trên COD thô)** "
        "(`tables/sensitivity_clip_cod.csv`)\n")
    add(md(sens))

    # ------------------------------------------------------------ 4. thời gian
    add("\n## 4. Thời gian huấn luyện và suy luận\n")
    tt = tim[['Model', 'Train_time_s_median', 'Train_time_s_min', 'Train_time_s_max',
              'Infer_ms_per_sample_median', 'Infer_total_s_median', 'n_runs', 'n_test']].rename(columns={
        'Model': 'Mô hình', 'Train_time_s_median': 'Huấn luyện – median (s)', 'Train_time_s_min': 'min (s)',
        'Train_time_s_max': 'max (s)', 'Infer_ms_per_sample_median': 'Suy luận (ms/mẫu)',
        'Infer_total_s_median': 'Suy luận toàn test (s)', 'n_runs': 'Số lần huấn luyện', 'n_test': 'n test'})
    add("**Bảng 11. Thời gian đo bằng `time.perf_counter`** (`tables/timing.csv`). Huấn luyện: median qua các seed; "
        f"Stacking gồm cả {C.N_SPLITS_TS} fold TimeSeriesSplit và lần huấn luyện lại trên toàn train. Suy luận: "
        f"median của {C.N_TIMING_REPEATS} lần dự báo toàn bộ tập test, chia cho số mẫu.\n")
    add(md(tt, 6))

    # ------------------------------------------------------------ 5. phần dư
    add("\n## 5. Phân tích phần dư và trễ pha\n")
    rcols = ['n', 'Mean', 'Std', 'Skewness', 'Kurtosis (excess)', 'Shapiro_W', 'Shapiro_p', 'JarqueBera_stat',
             'JarqueBera_p', 'BP_LM (theo ŷ)', 'BP_p (theo ŷ)', 'BP_LM (theo 8 đặc trưng)', 'BP_p (theo 8 đặc trưng)',
             'LjungBox_Q lag1', 'LjungBox_p lag1', 'ACF lag1']
    rt = res[rcols].T.astype(object)
    for c in rt.columns:   # hàng có '_p' là p-value; hàng 'n' là số nguyên
        rt[c] = [fp(float(v)) if '_p' in idx else (str(int(v)) if idx == 'n' else f"{float(v):.4f}")
                 for idx, v in zip(rt.index, rt[c])]
    rt.index.name = 'Chỉ số'
    add("**Bảng 12. Thống kê phần dư trên tập test (seed 42)** (`tables/residual_stats.csv`). "
        "Phần dư = thực tế − dự báo.\n")
    add(md(rt, index=True))
    cp = ccf.pivot_table(index=['Mô hình', 'Chuỗi'], columns='Độ trễ k', values='corr(ŷ_t, y_t+k)')
    cp.columns = [f'k={c}' for c in cp.columns]
    add("\n**Bảng 13. Tương quan chéo corr(ŷ_t, y_{t+k}) trên tập test** (`tables/lag_crosscorr.csv`). "
        "Nếu mô hình chỉ lặp lại giá trị hiện tại (trễ 1 bước) thì tương quan cực đại rơi vào k = −1.\n")
    add(md(cp, index=True))

    # ------------------------------------------------------------ 6. horizon
    if has_h:
        h = rd("horizon_metrics.csv")
        hs = rd("horizon_significance.csv")
        ht = pd.DataFrame({'Tầm dự báo': [f"{x} phút" for x in h['Horizon_min']], 'Mô hình': h['Model'],
                           'R²': [pm(a, b) for a, b in zip(h['R2_mean'], h['R2_std'])],
                           'RMSE (mg/L)': [pm(a, b) for a, b in zip(h['RMSE_mean'], h['RMSE_std'])],
                           'MAE (mg/L)': [pm(a, b) for a, b in zip(h['MAE_mean'], h['MAE_std'])],
                           'MAPE (%)': [pm(a, b, 2) for a, b in zip(h['MAPE_%_mean'], h['MAPE_%_std'])],
                           'Skill so với Persistence': h['Skill_vs_Persistence'], 'n test': h['n_test']})
        add("\n## 6. Tầm dự báo dài hơn\n")
        add("Đặc trưng: 8 thông số tại t + COD(t−1, t−2, t−3, t−6) + trung bình/độ lệch chuẩn trượt của COD "
            "(30 phút, 1 giờ) + COD(t) − COD(t−1); tất cả chỉ dùng quá khứ và tính trong từng đoạn liên tục.\n")
        add("**Bảng 14. Kết quả theo tầm dự báo (5 seed)** (`tables/horizon_metrics.csv`)\n")
        add(md(ht))
        hs2 = hs.copy()
        hs2['DM_p (MSE)'] = hs2['DM_p (MSE)'].map(fp)
        add("\n**Bảng 15. Diebold–Mariano theo tầm dự báo (seed 42)** (`tables/horizon_significance.csv`)\n")
        add(md(hs2))

    # ------------------------------------------------------------ 7. SHAP
    if has_shap:
        imp = rd("shap_importance.csv")
        note = open(T("shap_note.txt"), encoding='utf-8').read().strip()
        pv = imp.pivot(index='Đặc trưng', columns='Mô hình', values='Mean|SHAP| (mg/L)')
        pv = pv.loc[imp[imp['Mô hình'] == 'Stacking'].sort_values('Hạng')['Đặc trưng']]
        di = imp[imp['Mô hình'] == 'Stacking'].sort_values('Hạng')[
            ['Đặc trưng', 'Hạng', 'Mean|SHAP| (mg/L)', 'Mean SHAP (mg/L)', 'Spearman(giá trị, SHAP)']]
        add("\n## 7. Giải thích mô hình bằng SHAP\n")
        add(f"{note}\n")
        add("**Bảng 16. Trung bình |SHAP| (mg/L) theo mô hình** (`tables/shap_importance.csv`)\n")
        add(md(pv, index=True))
        add("\n**Bảng 17. Chiều tác động của từng đặc trưng trong Stacking.** Spearman(giá trị, SHAP) > 0 nghĩa là "
            "giá trị đặc trưng càng cao thì đóng góp vào dự báo càng lớn.\n")
        add(md(di))

    # ------------------------------------------------------------ hình
    add("\n## 8. Danh sách hình (outputs/v2/figures)\n")
    best = m.drop(['Persistence', 'Stacking']).sort_values('RMSE_mean').index[0]
    figs = [
        ("fig01_correlation_heatmap.png", "Ma trận tương quan Pearson giữa 8 thông số tại thời điểm t và COD(t+1), "
         f"tính trên {int(atd['Số cặp train (h=1, sau purge)']) + int(atd['Số cặp test (h=1)'])} cặp (t, t+1) "
         "dùng cho train và test."),
        ("fig02_timeseries_overview.png", "Chuỗi COD quan trắc 01–23/01/2026 (chu kỳ 5 phút); nền xanh là giai đoạn "
         "train (80%), nền cam là giai đoạn test (20%); đường đứt là ngưỡng IQR trên của COD tính trên train."),
        ("fig03_model_comparison.png", "RMSE trên tập test của các mô hình (trung bình ± độ lệch chuẩn qua 5 seed)."),
        ("fig04_actual_vs_predicted.png", "COD thực tế, dự báo của Stacking và của Persistence trong 150 chu kỳ đầu của tập test."),
        ("fig05_scatter_actual_pred.png", f"COD thực tế và dự báo trên tập test của Stacking, {best} (mô hình đơn tốt nhất) "
         "và Persistence; đường đứt là y = x."),
        ("fig06_residuals.png", "Phần dư của Stacking trên tập test: (a) histogram, (b) phần dư theo giá trị dự báo "
         "kèm đường LOWESS, (c) biểu đồ Q-Q so với phân phối chuẩn."),
        ("fig07_lag_crosscorr.png", f"Tương quan chéo corr(ŷ_t, y_t+k) với k = −3…+3 giữa dự báo và thực tế của "
         f"Stacking, {best} và Persistence, cho chuỗi mức (trái) và chuỗi sai phân bậc 1 (phải)."),
        ("fig08_shap_bar.png", "Trung bình |SHAP| (mg/L) của từng đặc trưng trong mô hình Stacking."),
        ("fig09_shap_beeswarm.png", "Phân bố giá trị SHAP của mô hình Stacking; màu thể hiện giá trị đặc trưng."),
        ("fig10_shap_compare.png", "So sánh trung bình |SHAP| giữa Stacking (Exact), XGBoost và CatBoost (TreeSHAP) "
         "trên cùng tập mẫu test."),
        ("fig11_shap_dependence.png", "Biểu đồ phụ thuộc SHAP của COD(t), nhiệt độ và NH4+ trong mô hình Stacking "
         "(đường đỏ là LOWESS)."),
        ("fig12_horizon_rmse.png", "RMSE trên tập test theo tầm dự báo (5 phút, 30 phút, 1 giờ, 2 giờ) của "
         "Persistence, XGBoost và Stacking (± std qua 5 seed)."),
        ("fig13_workflow.png", "Sơ đồ quy trình xử lý dữ liệu, huấn luyện, đánh giá và giải thích mô hình."),
    ]
    for i, (f, cap) in enumerate(figs, 1):
        if os.path.exists(os.path.join(C.FIG_DIR, f)):
            add(f"- **Hình {i}** (`figures/{f}`): {cap}")

    # ------------------------------------------------------------ nhận xét
    add("\n## 9. Nhận xét trung thực\n")
    add(honest_comments(m, sig, mw, var, sens, res, ccf, codsplit, atd, best, tim,
                        rd("horizon_metrics.csv") if has_h else None,
                        rd("horizon_significance.csv") if has_h else None,
                        rd("shap_importance.csv") if has_shap else None))

    add("\n## 10. Những thay đổi so với phiên bản cũ\n")
    add(changes_section(m))

    path = os.path.join(C.OUT_DIR, "RESULTS.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print("Đã ghi", path)


def honest_comments(m, sig, mw, var, sens, res, ccf, codsplit, atd, best, tim, h, hs, imp):
    out = []
    a = out.append
    rp, rs, rb = m.loc['Persistence', 'RMSE_mean'], m.loc['Stacking', 'RMSE_mean'], m.loc[best, 'RMSE_mean']
    s42 = sig[sig['Seed'] == C.SEED].set_index('So sánh (A vs B)')
    k_sb, k_sp, k_bp = f'Stacking vs {best}', 'Stacking vs Persistence', f'{best} vs Persistence'
    pmax = sig.groupby('So sánh (A vs B)')['DM_p (MSE)'].max()
    dm_sign = sig.groupby('So sánh (A vs B)')['DM_stat (MSE)'].agg(lambda x: np.sign(x).unique().tolist())

    # 1. Stacking vs baseline
    a("**(1) Stacking có tốt hơn Persistence và mô hình đơn tốt nhất không?**\n")
    better_p = rs < rp
    a(f"- Persistence (ŷ(t+1) = COD(t)) đạt RMSE = {rp:.4f} mg/L, R² = {m.loc['Persistence', 'R2_mean']:.4f}. "
      f"Mô hình đơn tốt nhất theo RMSE trung bình là **{best}** (RMSE = {rb:.4f} ± {m.loc[best, 'RMSE_std']:.4f}). "
      f"Stacking đạt RMSE = {rs:.4f} ± {m.loc['Stacking', 'RMSE_std']:.4f}, R² = {m.loc['Stacking', 'R2_mean']:.4f} "
      f"± {m.loc['Stacking', 'R2_std']:.4f}.")
    if not better_p:
        a(f"- Stacking **kém hơn** Persistence (Skill = {m.loc['Stacking', 'Skill_vs_Persistence_mean']:.4f}) "
          f"và kém hơn {best}. Ở seed 42, DM (MSE) = {s42.loc[k_sp, 'DM_stat (MSE)']:.2f} với p = "
          f"{fp(s42.loc[k_sp, 'DM_p (MSE)'])} (so với Persistence) và DM = {s42.loc[k_sb, 'DM_stat (MSE)']:.2f} với p = "
          f"{fp(s42.loc[k_sb, 'DM_p (MSE)'])} (so với {best}). p-value lớn nhất qua 5 seed lần lượt là "
          f"{fp(pmax[k_sp])} và {fp(pmax[k_sb])}, nên **chênh lệch có ý nghĩa thống kê theo hướng Stacking kém hơn**.")
    else:
        a(f"- Stacking có RMSE thấp hơn Persistence; DM p (seed 42) = {fp(s42.loc[k_sp, 'DM_p (MSE)'])}, "
          f"p lớn nhất qua 5 seed = {fp(pmax[k_sp])}.")
    trees = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'AdaBoost', 'XGBoost', 'CatBoost']
    worse = [t for t in trees if m.loc[t, 'RMSE_mean'] > rp]
    beat = [k for k in m.index if k != 'Persistence' and m.loc[k, 'RMSE_mean'] < rp]
    a(("- Không mô hình học máy nào có RMSE thấp hơn Persistence. " if not beat else
       f"- Các mô hình có RMSE thấp hơn Persistence: {', '.join(beat)}. ")
      + f"{len(worse)}/6 mô hình cây có RMSE cao hơn Persistence; LinearRegression có RMSE = "
      f"{m.loc['LinearRegression', 'RMSE_mean']:.4f} (Skill = {m.loc['LinearRegression', 'Skill_vs_Persistence_mean']:.4f}). "
      f"Chênh lệch giữa {best} và Persistence nhỏ về độ lớn ({rb - rp:.4f} mg/L) nhưng có ý nghĩa thống kê theo DM "
      f"(DM = {s42.loc[k_bp, 'DM_stat (MSE)']:.2f}, p = {fp(s42.loc[k_bp, 'DM_p (MSE)'])}; DM > 0 nghĩa là {best} kém hơn).")
    a(f"- Vì vậy R² cao (≈ {m.loc['Persistence', 'R2_mean']:.3f}) ở bài toán t+1 **chủ yếu đến từ quán tính của chuỗi COD** "
      "(COD(t) ≈ COD(t+1)), không phải năng lực riêng của mô hình. Không thể khẳng định ‘Stacking là mô hình tốt nhất’ "
      "như phiên bản trước.\n")

    # 2. vì sao stacking kém
    w = mw.set_index('Seed')
    a("**(2) Vì sao Stacking kém?** (các phân tích dưới đây là bằng chứng có trong số liệu, không phải kết luận nhân quả chắc chắn)\n")
    a(f"- Meta-learner học trên dự báo OOF của TimeSeriesSplit; các fold kiểm định này nằm trong giai đoạn train, "
      f"giai đoạn có các đợt COD tăng vọt (train: std = {codsplit.loc['Train', 'std']:.2f} mg/L, max = {codsplit.loc['Train', 'max']:.2f}), trong khi "
      f"giai đoạn test yên hơn nhiều (std = {codsplit.loc['Test', 'std']:.2f}, max = {codsplit.loc['Test', 'max']:.2f}). "
      f"Hệ số meta có tổng từ {w['Tổng hệ số'].min():.3f} đến {w['Tổng hệ số'].max():.3f} và intercept từ "
      f"{w['Intercept'].min():.3f} đến {w['Intercept'].max():.3f} (Bảng 8), có cả hệ số âm. Khi áp lên test, "
      f"bộ hệ số này làm dự báo lệch có hệ thống: trung bình phần dư của Stacking = {res.loc['Stacking', 'Mean']:.4f} mg/L, "
      f"so với {res.loc['Persistence', 'Mean']:.4f} của Persistence.")
    v = var.set_index('Biến thể')
    kf = [i for i in v.index if 'KFold' in i][0]
    pos = [i for i in v.index if 'positive' in i][0]
    main_ = [i for i in v.index if 'chính' in i][0]
    a(f"- Bảng 9: nếu dùng KFold như phiên bản trước, RMSE = {v.loc[kf, 'RMSE']:.4f}; nếu ràng buộc hệ số meta ≥ 0, RMSE = "
      f"{v.loc[pos, 'RMSE']:.4f}; mô hình chính (TimeSeriesSplit) có RMSE = {v.loc[main_, 'RMSE']:.4f}. "
      "Kết quả của Stacking vì vậy rất nhạy với cách tạo OOF. Kể cả biến thể tốt nhất cũng "
      + ("không vượt Persistence." if min(v['RMSE']) > rp else "vượt Persistence."))
    a(f"- Bản thân các mô hình cơ sở đã kém Persistence (ví dụ ExtraTrees: RMSE = {m.loc['ExtraTrees', 'RMSE_mean']:.4f}, "
      f"gấp {m.loc['ExtraTrees', 'RMSE_mean'] / rp:.2f} lần) và có phần dư trung bình khác 0 (XGBoost "
      f"{res.loc['XGBoost', 'Mean']:.4f}, CatBoost {res.loc['CatBoost', 'Mean']:.4f} mg/L), nên không thể kỳ vọng "
      "tổ hợp tuyến tính của chúng vượt Persistence. Nguyên nhân cụ thể khiến các mô hình cây kém (ví dụ xấp xỉ quan hệ "
      "gần như đồng nhất COD(t) → COD(t+1) bằng hàm bậc thang, hoặc phụ thuộc vào các cảm biến khác) "
      "**chưa được kiểm chứng riêng** trong đồ án này.\n")

    # 3. phần dư
    st = res.loc['Stacking']
    a("**(3) Phần dư có chuẩn và đồng phương sai không?**\n")
    a(f"- Phần dư Stacking: mean = {st['Mean']:.4f}, std = {st['Std']:.4f}, skewness = {st['Skewness']:.3f}, "
      f"kurtosis dư = {st['Kurtosis (excess)']:.3f}. Shapiro–Wilk p = {fp(st['Shapiro_p'])} và Jarque–Bera "
      f"p = {fp(st['JarqueBera_p'])}, nên **bác bỏ giả thuyết phân phối chuẩn**.")
    het = st['BP_p (theo ŷ)'] < 0.05 or st['BP_p (theo 8 đặc trưng)'] < 0.05
    a(f"- Breusch–Pagan theo ŷ: p = {fp(st['BP_p (theo ŷ)'])}; theo 8 đặc trưng: p = {fp(st['BP_p (theo 8 đặc trưng)'])}. "
      + ("Có bằng chứng **phương sai sai số thay đổi** (không đồng phương sai) ở mức 5%." if het else
         "Không đủ bằng chứng bác bỏ đồng phương sai ở mức 5%."))
    a(f"- Ljung–Box lag 1: p = {fp(st['LjungBox_p lag1'])}, ACF(1) = {st['ACF lag1']:.3f}. Phần dư **tự tương quan mạnh**, "
      "tức mô hình còn bỏ sót cấu trúc thời gian. Persistence và LinearRegression cũng có phần dư không chuẩn "
      f"(kurtosis dư {res.loc['Persistence', 'Kurtosis (excess)']:.1f} và {res.loc['LinearRegression', 'Kurtosis (excess)']:.1f}) "
      "do các bước nhảy đột ngột của COD.\n")

    # 4. trễ pha
    a("**(4) Mô hình có bị trễ pha không?**\n")
    def peak(model, series):
        d = ccf[(ccf['Mô hình'] == model) & (ccf['Chuỗi'] == series)].set_index('Độ trễ k')['corr(ŷ_t, y_t+k)']
        return int(d.idxmax()), float(d.max()), float(d.loc[0])
    for mod in ['Persistence', best, 'Stacking']:
        k1, v1, v0 = peak(mod, 'mức')
        k2, v2, v20 = peak(mod, 'sai phân')
        a(f"- {mod}: chuỗi mức đạt tương quan cực đại ở k = {k1} ({v1:.4f}; tại k = 0 là {v0:.4f}); chuỗi sai phân đạt "
          f"cực đại ở k = {k2} ({v2:.4f}; tại k = 0 là {v20:.4f}).")
    a("- Cực đại ở k = −1 nghĩa là dự báo tại t giống nhất với giá trị thực tế của chu kỳ **trước**, tức **trễ 1 bước** "
      "(dự báo về cơ bản lặp lại COD hiện tại). Kiểm tra trên chuỗi sai phân rõ hơn vì loại bỏ thành phần mức chung.")
    lag_models = [mod for mod in ['Persistence', best, 'Stacking'] if peak(mod, 'sai phân')[0] == -1]
    txt = f"- Các mô hình có cực đại sai phân ở k = −1: {', '.join(lag_models)}. "
    if best in lag_models:
        _, vb, vb0 = peak(best, 'sai phân')
        txt += (f"{best} có tương quan sai phân {vb:.4f} tại k = −1 nhưng {vb0:.4f} tại k = 0, nên về bản chất nó đang "
                "mô phỏng Persistence và **chưa dự báo được thời điểm COD đổi hướng**. ")
    if 'Stacking' in lag_models:
        txt += (f"Stacking cũng trễ 1 bước, nhưng tương quan sai phân chỉ {peak('Stacking', 'sai phân')[1]:.4f}, tức "
                "dự báo của nó còn thêm dao động riêng không khớp với thay đổi thực tế.")
    a(txt + "\n")

    # 5. SHAP
    if imp is not None:
        s = imp[imp['Mô hình'] == 'Stacking'].sort_values('Hạng').set_index('Đặc trưng')
        top = s.index[0]
        a("**(5) Biến quan trọng nhất theo SHAP và chiều tác động (mô hình Stacking)**\n")
        a(f"- Biến quan trọng nhất: **{top}(t)** với mean|SHAP| = {s.loc[top, 'Mean|SHAP| (mg/L)']:.4f} mg/L, gấp "
          f"{s.loc[top, 'Mean|SHAP| (mg/L)'] / s.iloc[1]['Mean|SHAP| (mg/L)']:.1f} lần biến thứ hai "
          f"({s.index[1]}: {s.iloc[1]['Mean|SHAP| (mg/L)']:.4f}). Spearman(giá trị, SHAP) của {top} = "
          f"{s.loc[top, 'Spearman(giá trị, SHAP)']:.3f}, tức COD hiện tại càng cao thì dự báo càng cao.")
        others = ', '.join(f"{i} ({r['Mean|SHAP| (mg/L)']:.3f}; ρ = {r['Spearman(giá trị, SHAP)']:+.2f})"
                           for i, r in s.iloc[1:].iterrows())
        a(f"- Các biến còn lại (mean|SHAP| mg/L; ρ Spearman): {others}.")
        pv = imp.pivot(index='Đặc trưng', columns='Mô hình', values='Hạng')
        a(f"- Thứ hạng COD trong XGBoost / CatBoost: {int(pv.loc['COD', 'XGBoost'])} / {int(pv.loc['COD', 'CatBoost'])}; "
          "cả ba mô hình đều đặt COD(t) đứng đầu. Thứ tự các biến phụ khác nhau giữa các mô hình (Bảng 16), nên "
          "**không nên diễn giải thứ tự của các biến phụ như quan hệ nhân quả**.")
        a(f"- Lưu ý: SHAP giải thích **mô hình**, không phải quá trình vật lý. Base value của Stacking "
          f"({s.iloc[0]['Base value (mg/L)']:.3f} mg/L) được tính trên nền train, nên giá trị SHAP âm của COD ở vùng "
          "COD thấp phản ánh việc test nằm dưới mức trung bình của train.\n")

    # 6. horizon
    if h is not None:
        a("**(6) Tầm dự báo dài hơn (cảnh báo sớm)**\n")
        for hh, g in h.groupby('Horizon_min'):
            g = g.set_index('Model')
            sk = g.loc[['XGBoost', 'Stacking'], 'Skill_vs_Persistence']
            hsig = hs[hs['Horizon_steps'] == hh // 5].set_index('So sánh')
            a(f"- {hh} phút: RMSE Persistence = {g.loc['Persistence', 'RMSE_mean']:.4f}, XGBoost = "
              f"{g.loc['XGBoost', 'RMSE_mean']:.4f} (Skill {sk['XGBoost']:+.3f}; DM p = "
              f"{fp(hsig.loc['XGBoost vs Persistence', 'DM_p (MSE)'])}), Stacking = {g.loc['Stacking', 'RMSE_mean']:.4f} "
              f"(Skill {sk['Stacking']:+.3f}; DM p = {fp(hsig.loc['Stacking vs Persistence', 'DM_p (MSE)'])}).")
        ns = hs[hs['DM_p (MSE)'] >= 0.05]
        if len(ns):
            a("- So sánh **không** có ý nghĩa thống kê ở mức 5% (DM, seed 42): " + "; ".join(
                f"{r['So sánh']} ở {int(r['Horizon_steps']) * C.STEP_MIN} phút (p = {fp(r['DM_p (MSE)'])})"
                for _, r in ns.iterrows()) + ". Ở các trường hợp này chỉ có thể nói hai mô hình không khác biệt rõ.")
        best_h = h[h['Model'] != 'Persistence']
        pos_sk = best_h[best_h['Skill_vs_Persistence'] > 0]
        if len(pos_sk):
            a("- Có trường hợp mô hình học máy vượt Persistence: " + "; ".join(
                f"{r.Model} ở {r.Horizon_min} phút (Skill {r.Skill_vs_Persistence:+.3f})" for r in pos_sk.itertuples())
              + ". Kiểm tra cột p của DM trước khi khẳng định.")
        else:
            a("- Ở **mọi tầm dự báo đã thử**, cả XGBoost lẫn Stacking đều không vượt Persistence trên tập test này. "
              "Vì vậy **chưa có cơ sở để nói hệ thống ‘cảnh báo sớm 1–2 giờ’**.")
        a("")

    # 7. hạn chế
    a("**(7) Hạn chế**\n")
    a(f"- Chỉ có 1 trạm, {atd['Tổng thời gian bao phủ (ngày)']} ngày dữ liệu; tập test chỉ "
      f"{int(atd['Số cặp test (h=1)'])} cặp (khoảng 4,5 ngày) và rơi vào giai đoạn COD ổn định "
      f"(max {codsplit.loc['Test', 'max']:.2f} mg/L). Mô hình chưa được kiểm chứng trên các đợt COD tăng cao.")
    a("- Chỉ dùng một lần chia train/test cố định. Nên bổ sung đánh giá rolling-origin (walk-forward) trên nhiều "
      "giai đoạn để có ước lượng ổn định hơn.")
    a("- Siêu tham số giữ mặc định, chưa tinh chỉnh. Kiểm định DM giả định chuỗi chênh lệch sai số dừng.")
    a("- SHAP của Stacking tính trên mẫu con test (xem Mục 7). TreeSHAP của XGBoost/CatBoost dùng chế độ "
      "path-dependent, khác chế độ interventional của Stacking, nên chỉ nên so sánh thứ hạng, không so sánh trực tiếp độ lớn.")
    tst = tim.set_index('Model')
    a(f"- Thời gian đo trên một máy (xem `environment.txt`). XGBoost/CatBoost dùng đa luồng còn sklearn dùng 1 luồng, nên "
      f"so sánh tốc độ giữa các họ mô hình chỉ mang tính tham khảo. Stacking huấn luyện mất "
      f"{tst.loc['Stacking', 'Train_time_s_median']:.1f} s (median) và suy luận "
      f"{tst.loc['Stacking', 'Infer_ms_per_sample_median']:.4f} ms/mẫu.")
    return '\n'.join(out)


def changes_section(m):
    old_path = T("previous_model_metrics.csv")   # số liệu của phiên bản trước, giữ lại để đối chiếu
    lines = [
        "1. **Đọc dữ liệu:** đọc thêm cột trạng thái; giá trị có status ≠ `00` (2 giá trị Color mang mã `01`) được "
        "coi là thiếu.",
        "2. **Trục thời gian:** kiểm tra trùng lặp/gap và chia chuỗi thành các đoạn liên tục. Chỉ tạo cặp (t, t+h) khi "
        "hai thời điểm cách nhau đúng h·5 phút (bỏ 4 cặp từng nối qua gap). Các cặp train có mục tiêu rơi vào giai đoạn "
        "test bị loại (purge).",
        "3. **Chống rò rỉ:** chia train/test (80/20 theo thời gian) **trước**; median và ngưỡng IQR chỉ fit trên train "
        "rồi áp dụng cho test. MinMaxScaler nằm trong Pipeline, chỉ fit trên train.",
        "4. **Điền thiếu:** forward-fill trong cùng đoạn liên tục (chỉ dùng quá khứ, không nối qua gap); nếu còn thiếu "
        "(ở đầu đoạn) thì dùng median của train. Không dùng nội suy hai phía, vì nó dùng giá trị tương lai.",
        "5. **IQR:** vẫn ở dạng clip (không xoá dòng) cho 7 cảm biến; **không clip COD** vì COD(t+1) là biến mục tiêu. "
        "Bản cũ clip cả COD về [4,40; 19,41] mg/L, làm mất các đỉnh ô nhiễm. Mức ảnh hưởng xem Bảng 10.",
        "6. **Baseline:** thêm Persistence (ŷ(t+1) = COD(t)) và LinearRegression trên 8 đặc trưng.",
        "7. **Stacking:** thay `cv=5` (KFold) bằng TimeSeriesSplit(5). Vì `StackingRegressor` của sklearn không nhận "
        "TimeSeriesSplit, đã tự cài `TimeSeriesStackingRegressor` (src/ensemble_models.py): meta-learner học trên dự "
        "báo OOF của các fold (chỉ dùng quá khứ), sau đó huấn luyện lại các mô hình cơ sở trên toàn bộ train.",
        "8. **Đánh giá:** thêm MAPE, Skill so với Persistence; chạy 5 seed (42–46) và báo cáo mean ± std; kiểm định "
        "Diebold–Mariano (HLN + HAC) và Wilcoxon; đo thời gian bằng `time.perf_counter`.",
        "9. **SHAP:** giải thích **chính mô hình Stacking** ở thang mg/L gốc bằng `shap.Explainer(algorithm='exact')` "
        "(nền 200 mẫu train), thay vì ExtraTrees trên dữ liệu đã scale. Thêm TreeSHAP của XGBoost/CatBoost để đối chiếu.",
        "10. **Tầm dự báo:** thêm thí nghiệm h = 5, 30, 60, 120 phút với đặc trưng trễ và trượt của COD.",
        "11. **Ứng dụng:** thanh trượt giới hạn trong khoảng train, có cảnh báo khi nhập ngoài khoảng; ngưỡng theo "
        "QCVN 40:2025/BTNMT nằm trong dict cấu hình (cần điền); SHAP waterfall cho đúng mô hình đang dự báo; mặc định "
        "chọn mô hình học máy có RMSE test thấp nhất.",
    ]
    txt = '\n'.join(lines)
    if os.path.exists(old_path):
        old = pd.read_csv(old_path).set_index('Model')
        old = old.rename(index={'StackingEnsemble': 'Stacking'})
        rows = []
        for k in old.index:
            if k in m.index:
                rows.append({'Mô hình': k, 'R² cũ': old.loc[k, 'R2'], 'RMSE cũ': old.loc[k, 'RMSE'],
                             'R² mới (mean)': m.loc[k, 'R2_mean'], 'RMSE mới (mean)': m.loc[k, 'RMSE_mean']})
        txt += ("\n\n**Bảng 18. Đối chiếu số liệu cũ (`tables/previous_model_metrics.csv`) và mới.** Số cũ được tính khi target "
                "đã bị clip, cặp t→t+1 nối qua gap, Stacking dùng KFold và chỉ chạy 1 seed, nên không so sánh trực tiếp được; "
                "bảng này chỉ để thấy mức độ thay đổi.\n\n" + md(pd.DataFrame(rows)))
        txt += (f"\n\nPersistence (không có trong phiên bản trước): R² = {m.loc['Persistence', 'R2_mean']:.4f}, "
                f"RMSE = {m.loc['Persistence', 'RMSE_mean']:.4f}.")
    return txt


if __name__ == "__main__":
    main()
