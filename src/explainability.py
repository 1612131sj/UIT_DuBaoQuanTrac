"""
SHAP cho mô hình Stacking (thang mg/L gốc) + đối chiếu TreeSHAP của XGBoost/CatBoost.

- Stacking: shap.Explainer(algorithm='exact') với masker Independent, nền = 200 mẫu ngẫu nhiên từ train.
  Với 8 đặc trưng, thuật toán 'exact' duyệt đủ 2^8 = 256 tổ hợp nên cho giá trị Shapley chính xác
  (theo nghĩa interventional, so với tập nền), không phải xấp xỉ như PermutationExplainer.
  Đầu vào là giá trị gốc (mg/L, °C, ...) vì Pipeline đã bao gồm MinMaxScaler.
- XGBoost, CatBoost: TreeExplainer chế độ tree_path_dependent, dùng để đối chiếu thứ hạng đặc trưng.
- Khi tính SHAP, RF/ExtraTrees trong Stacking chạy n_jobs=-1 (chỉ tăng tốc, không đổi kết quả).
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm

try:
    from . import config as C
    from . import plot_style as S
    from .data_preprocessing import load_dataset
except ImportError:
    import config as C
    import plot_style as S
    from data_preprocessing import load_dataset

S.setup()
FEAT_LABEL = {c: f'{c} ({C.UNITS[c]})' if C.UNITS[c] != '-' else c for c in C.FEATURES}


def load():
    ds = load_dataset(h=1)
    models = {n: joblib.load(os.path.join(C.MODELS_DIR, f"{n}.pkl"))
              for n in ['Stacking', 'XGBoost', 'CatBoost']}
    bg = np.load(os.path.join(C.MODELS_DIR, "shap_background.npy"))
    return ds, models, bg


def compute(ds, models, bg):
    Xte = ds['X_test']
    n = len(Xte)
    rng = np.random.default_rng(C.SEED)
    if n > C.SHAP_TEST_N:
        idx = np.sort(rng.choice(n, C.SHAP_TEST_N, replace=False))
    else:
        idx = np.arange(n)
    X = Xte[idx]
    note = (f"SHAP tính trên {len(idx)}/{n} mẫu test "
            f"({'lấy mẫu ngẫu nhiên, seed 42' if len(idx) < n else 'toàn bộ test'}); "
            f"nền = {len(bg)} mẫu ngẫu nhiên từ train (seed 42).")
    print(note)

    out = {}
    # --- Stacking (exact Shapley, mg/L)
    st = models['Stacking']
    for est in st.named_steps['model'].estimators_:      # tăng tốc dự báo, không đổi kết quả
        if 'n_jobs' in est.get_params():
            est.set_params(n_jobs=-1)
    t0 = time.perf_counter()
    expl = shap.Explainer(st.predict, shap.maskers.Independent(bg, max_samples=len(bg)),
                          algorithm='exact', feature_names=C.FEATURES)
    sv = expl(X)
    t_st = time.perf_counter() - t0
    add_err = np.abs(sv.values.sum(1) + sv.base_values - st.predict(X)).max()
    out['Stacking'] = (sv.values, np.atleast_1d(sv.base_values)[0])
    print(f"Stacking exact SHAP: {t_st:.1f}s, sai số cộng tính max = {add_err:.2e}")

    # --- XGBoost, CatBoost (TreeSHAP, tree_path_dependent)
    for m in ['XGBoost', 'CatBoost']:
        pipe = models[m]
        te = shap.TreeExplainer(pipe.named_steps['model'], feature_perturbation='tree_path_dependent')
        out[m] = (np.asarray(te.shap_values(pipe.named_steps['scaler'].transform(X))),
                  float(np.atleast_1d(te.expected_value)[0]))

    # --- bảng tầm quan trọng + chiều tác động
    rows = []
    for m, (vals, base) in out.items():
        for j, c in enumerate(C.FEATURES):
            r = np.corrcoef(X[:, j], vals[:, j])[0, 1] if X[:, j].std() > 0 else np.nan
            rows.append({'Mô hình': m, 'Đặc trưng': c, 'Mean|SHAP| (mg/L)': np.abs(vals[:, j]).mean(),
                         'Mean SHAP (mg/L)': vals[:, j].mean(),
                         'Spearman(giá trị, SHAP)': pd.Series(X[:, j]).corr(pd.Series(vals[:, j]), method='spearman'),
                         'Pearson(giá trị, SHAP)': r, 'Base value (mg/L)': base})
    imp = pd.DataFrame(rows)
    imp['Hạng'] = imp.groupby('Mô hình')['Mean|SHAP| (mg/L)'].rank(ascending=False).astype(int)
    imp.to_csv(os.path.join(C.TAB_DIR, "shap_importance.csv"), index=False, encoding='utf-8-sig')
    with open(os.path.join(C.TAB_DIR, "shap_note.txt"), 'w', encoding='utf-8') as f:
        f.write(note + f"\nThời gian tính SHAP Stacking (exact): {t_st:.1f} s; "
                       f"sai số cộng tính lớn nhất: {add_err:.2e} mg/L\n")
    return X, out, note, imp


def fig08_bar(imp, n):
    d = imp[imp['Mô hình'] == 'Stacking'].sort_values('Mean|SHAP| (mg/L)')
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.45))
    ax.barh([FEAT_LABEL[c] for c in d['Đặc trưng']], d['Mean|SHAP| (mg/L)'], color=S.BLUE, height=0.6)
    for i, v in enumerate(d['Mean|SHAP| (mg/L)']):
        ax.text(v, i, f' {v:.3f}', va='center', fontsize=7.5, color=S.INK2)
    ax.set_xlabel('Trung bình |giá trị SHAP| (mg/L)')
    ax.set_title(f'Mức độ quan trọng của đặc trưng – mô hình Stacking (n = {n} mẫu test)')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, d['Mean|SHAP| (mg/L)'].max() * 1.15)
    S.save(fig, os.path.join(C.FIG_DIR, "fig08_shap_bar.png"))


def fig09_beeswarm(X, vals, base, n):
    ex = shap.Explanation(values=vals, base_values=np.full(len(vals), base), data=X,
                          feature_names=[FEAT_LABEL[c] for c in C.FEATURES])
    plt.figure(figsize=(S.WIDTH, S.WIDTH * 0.55))
    shap.plots.beeswarm(ex, max_display=8, show=False, color_bar_label='Giá trị đặc trưng',
                        plot_size=None)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_xlabel('Giá trị SHAP – tác động lên COD(t+1) dự báo (mg/L)')
    ax.set_title(f'Phân bố giá trị SHAP – mô hình Stacking (n = {n} mẫu test)', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    if len(fig.axes) > 1:   # colorbar: đổi chữ High/Low sang tiếng Việt, cỡ chữ đồng bộ
        cb = fig.axes[-1]
        cb.set_yticklabels(['Thấp', 'Cao'], fontsize=8)
        cb.yaxis.label.set_size(9)
    S.save(fig, os.path.join(C.FIG_DIR, "fig09_shap_beeswarm.png"))


def fig10_compare(imp):
    order = (imp[imp['Mô hình'] == 'Stacking'].sort_values('Mean|SHAP| (mg/L)', ascending=False)['Đặc trưng'])
    models = ['Stacking', 'XGBoost', 'CatBoost']
    labels = {'Stacking': 'Stacking (Exact, interventional)', 'XGBoost': 'XGBoost (TreeSHAP, path-dependent)',
              'CatBoost': 'CatBoost (TreeSHAP, path-dependent)'}
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.5))
    y = np.arange(len(order))
    w = 0.27
    for i, m in enumerate(models):
        d = imp[imp['Mô hình'] == m].set_index('Đặc trưng').loc[order]
        ax.barh(y + (i - 1) * w, d['Mean|SHAP| (mg/L)'], height=w,
                color=S.MODEL_COLORS[m], label=labels[m])
    ax.set_yticks(y, [FEAT_LABEL[c] for c in order])
    ax.invert_yaxis()
    ax.set_xlabel('Trung bình |giá trị SHAP| (mg/L)')
    ax.set_title('So sánh mức độ quan trọng đặc trưng giữa các mô hình (cùng mẫu test)')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', visible=False)
    S.save(fig, os.path.join(C.FIG_DIR, "fig10_shap_compare.png"))


def fig11_dependence(X, vals):
    feats = ['COD', 'Temp', 'NH4+']
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, S.WIDTH * 0.36))
    for ax, c in zip(axes, feats):
        j = C.FEATURES.index(c)
        ax.scatter(X[:, j], vals[:, j], s=6, alpha=0.5, color=S.BLUE, edgecolors='none')
        lw = sm.nonparametric.lowess(vals[:, j], X[:, j], frac=0.4)
        ax.plot(lw[:, 0], lw[:, 1], color=S.RED, lw=1.4, label='LOWESS')
        ax.axhline(0, color=S.INK, lw=0.7, ls='--')
        ax.set_xlabel(f'{c}(t) ({C.UNITS[c]})')
        ax.set_title(c)
    axes[0].set_ylabel('Giá trị SHAP (mg/L)')
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('Biểu đồ phụ thuộc SHAP – mô hình Stacking', fontsize=10)
    fig.tight_layout()
    S.save(fig, os.path.join(C.FIG_DIR, "fig11_shap_dependence.png"))


def main():
    ds, models, bg = load()
    X, out, note, imp = compute(ds, models, bg)
    vals, base = out['Stacking']
    fig08_bar(imp, len(X))
    fig09_beeswarm(X, vals, base, len(X))
    fig10_compare(imp)
    fig11_dependence(X, vals)
    print(imp.pivot(index='Đặc trưng', columns='Mô hình', values='Mean|SHAP| (mg/L)').round(4))


if __name__ == "__main__":
    main()
