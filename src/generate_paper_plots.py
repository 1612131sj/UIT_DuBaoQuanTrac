"""
Hình cho báo cáo (trừ phần dư và SHAP):
fig01 heatmap tương quan, fig02 chuỗi COD, fig03 so sánh RMSE, fig04 thực tế vs dự báo,
fig05 scatter, fig07 tương quan chéo theo độ trễ, fig12 RMSE theo horizon, fig13 sơ đồ quy trình.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

try:
    from . import config as C
    from . import plot_style as S
    from .data_preprocessing import load_dataset
except ImportError:
    import config as C
    import plot_style as S
    from data_preprocessing import load_dataset

S.setup()
FIG = lambda name: os.path.join(C.FIG_DIR, name)


def best_single_model():
    m = pd.read_csv(os.path.join(C.TAB_DIR, "model_metrics.csv"))
    s = m[~m['Model'].isin(['Persistence', 'Stacking'])].sort_values('RMSE_mean')
    return s.iloc[0]['Model']


# ---------------------------------------------------------------- fig01
def fig01_correlation(ds):
    df = pd.concat([ds['train_df'], ds['test_df']])
    cols = C.FEATURES + ['y_target']
    corr = df[cols].corr(method='pearson')
    labels = [f'{c}(t)' for c in C.FEATURES] + ['COD(t+1)']
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.82))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, linecolor='white', annot_kws={'size': 7},
                cbar_kws={'label': 'Hệ số tương quan Pearson', 'shrink': 0.8},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Ma trận tương quan Pearson ({len(df)} cặp t → t+1 dùng cho train và test)')
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    ax.grid(False)
    S.save(fig, FIG("fig01_correlation_heatmap.png"))


# ---------------------------------------------------------------- fig02
def fig02_timeseries(ds):
    clean, cutoff = ds['clean'], ds['cutoff']
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.42))
    t0, t1 = clean['Timestamp'].min(), clean['Timestamp'].max()
    ax.axvspan(t0, cutoff, color=S.BLUE, alpha=0.07, lw=0, label='Giai đoạn train (80%)')
    ax.axvspan(cutoff, t1, color=S.ORANGE, alpha=0.12, lw=0, label='Giai đoạn test (20%)')
    for i, (_, seg) in enumerate(clean.groupby('segment')):   # không nối đường qua gap
        ax.plot(seg['Timestamp'], seg['COD_raw'], color=S.INK, lw=0.6,
                label='COD quan trắc' if i == 0 else None)
    ax.axhline(ds['prep'].upper_['COD'], color=S.RED, lw=0.9, ls='--',
               label='Ngưỡng IQR trên của COD (tính trên train)')
    ax.set_xlabel('Thời gian (tháng 01/2026)')
    ax.set_ylabel('COD (mg/L)')
    ax.set_title('Chuỗi COD toàn bộ giai đoạn quan trắc (chu kỳ 5 phút)')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d/%m'))
    ax.set_xlim(t0, t1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    S.save(fig, FIG("fig02_timeseries_overview.png"))


# ---------------------------------------------------------------- fig03
def fig03_model_comparison():
    m = pd.read_csv(os.path.join(C.TAB_DIR, "model_metrics.csv"))
    m = m.sort_values('RMSE_mean', ascending=True)
    colors = [S.ORANGE if n == 'Persistence' else S.BLUE if n == 'Stacking' else S.MUTED
              for n in m['Model']]
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.5))
    y = np.arange(len(m))
    ax.barh(y, m['RMSE_mean'], xerr=m['RMSE_std'], color=colors, height=0.6,
            error_kw={'ecolor': S.INK, 'elinewidth': 1, 'capsize': 3})
    for yi, (v, s) in enumerate(zip(m['RMSE_mean'], m['RMSE_std'])):
        ax.text(v + s + m['RMSE_mean'].max() * 0.01, yi, f'{v:.4f}',
                va='center', fontsize=7.5, color=S.INK2)
    ax.set_yticks(y, m['Model'])
    ax.invert_yaxis()
    ax.set_xlabel('RMSE trên tập test (mg/L) — trung bình ± độ lệch chuẩn qua 5 seed')
    ax.set_title('So sánh RMSE các mô hình dự báo COD(t+1)')
    ax.set_xlim(0, (m['RMSE_mean'] + m['RMSE_std']).max() * 1.15)
    ax.grid(axis='y', visible=False)
    S.save(fig, FIG("fig03_model_comparison.png"))


# ---------------------------------------------------------------- fig04
def fig04_actual_vs_pred(pr, n=150):
    p = pr.iloc[:n]
    x = np.arange(1, len(p) + 1)
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.45))
    ax.plot(x, p['y_true'], color=S.INK, lw=1.6, label='Thực tế')
    ax.plot(x, p['Stacking'], color=S.BLUE, lw=1.4, label='Stacking')
    ax.plot(x, p['Persistence'], color=S.ORANGE, lw=1.2, ls='--', label='Persistence')
    ax.set_xlabel(f'Chu kỳ 5 phút trong tập test (bắt đầu {p["Timestamp_target"].iloc[0]:%d/%m/%Y %H:%M})')
    ax.set_ylabel('COD (mg/L)')
    ax.set_title(f'COD thực tế và dự báo t+1 trên {len(p)} chu kỳ đầu của tập test')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=3)
    S.save(fig, FIG("fig04_actual_vs_predicted.png"))


# ---------------------------------------------------------------- fig05
def fig05_scatter(pr, best):
    names = ['Stacking', best, 'Persistence']
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, S.WIDTH * 0.36), sharex=True, sharey=True)
    lo = min(pr['y_true'].min(), pr[names].min().min()) - 0.3
    hi = max(pr['y_true'].max(), pr[names].max().max()) + 0.3
    for ax, n in zip(axes, names):
        ax.scatter(pr['y_true'], pr[n], s=6, alpha=0.45, color=S.MODEL_COLORS.get(n, S.AQUA),
                   edgecolors='none')
        ax.plot([lo, hi], [lo, hi], color=S.INK, lw=0.9, ls='--', label='y = x')
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect('equal')
        ax.set_title(n)
        ax.set_xlabel('COD thực tế (mg/L)')
    axes[0].set_ylabel('COD dự báo (mg/L)')
    axes[0].legend(loc='upper left', frameon=False)
    fig.suptitle(f'COD thực tế và dự báo t+1 trên tập test (n = {len(pr)})', fontsize=10)
    fig.tight_layout()
    S.save(fig, FIG("fig05_scatter_actual_pred.png"))


# ---------------------------------------------------------------- fig07
def lag_ccf(pr, col, lags=range(-3, 4), diff=False):
    """corr(ŷ_t, y_{t+k}) tính trong từng đoạn liên tục; diff=True dùng sai phân bậc 1."""
    out = []
    for k in lags:
        a_all, b_all = [], []
        for _, seg in pr.groupby('segment'):
            yh, y = seg[col].to_numpy(), seg['y_true'].to_numpy()
            if diff:
                yh, y = np.diff(yh), np.diff(y)
            if k >= 0:
                a, b = yh[:len(yh) - k], y[k:]
            else:
                a, b = yh[-k:], y[:len(y) + k]
            a_all.append(a), b_all.append(b)
        out.append(np.corrcoef(np.concatenate(a_all), np.concatenate(b_all))[0, 1])
    return np.array(out)


def fig07_lag_crosscorr(pr, best):
    lags = np.arange(-3, 4)
    names = ['Stacking', best, 'Persistence']
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH, S.WIDTH * 0.4))
    w = 0.26
    for i, n in enumerate(names):
        for ax, diff in zip(axes, [False, True]):
            r = lag_ccf(pr, n, lags, diff=diff)
            ax.bar(lags + (i - 1) * w, r, width=w, color=S.MODEL_COLORS.get(n, S.AQUA), label=n)
            rows += [{'Mô hình': n, 'Chuỗi': 'sai phân' if diff else 'mức', 'Độ trễ k': int(k),
                      'corr(ŷ_t, y_t+k)': float(v)} for k, v in zip(lags, r)]
    axes[0].set_title('Chuỗi mức COD')
    axes[1].set_title('Chuỗi sai phân bậc 1 (ΔCOD)')
    for ax in axes:
        ax.set_xticks(lags)
        ax.set_xlabel('Độ trễ k (chu kỳ 5 phút)')
        ax.axhline(0, color=S.INK, lw=0.6)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Tương quan corr(ŷ$_t$, y$_{t+k}$)')
    lv = [r['corr(ŷ_t, y_t+k)'] for r in rows if r['Chuỗi'] == 'mức']
    axes[0].set_ylim(np.floor(min(lv) * 20) / 20 - 0.05, 1.0)   # phóng to vùng giá trị để thấy chênh lệch
    fig.suptitle('Tương quan chéo giữa dự báo và thực tế theo độ trễ (tập test)', fontsize=10)
    fig.tight_layout()
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    S.save(fig, FIG("fig07_lag_crosscorr.png"))
    pd.DataFrame(rows).to_csv(os.path.join(C.TAB_DIR, "lag_crosscorr.csv"), index=False,
                              encoding='utf-8-sig')


# ---------------------------------------------------------------- fig12
def fig12_horizon():
    path = os.path.join(C.TAB_DIR, "horizon_metrics.csv")
    if not os.path.exists(path):
        print("  (bỏ qua fig12: chưa có horizon_metrics.csv)")
        return
    h = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 0.45))
    markers = {'Persistence': 's', 'XGBoost': '^', 'Stacking': 'o'}
    for n, g in h.groupby('Model', sort=False):
        ax.errorbar(g['Horizon_min'], g['RMSE_mean'], yerr=g['RMSE_std'], marker=markers.get(n, 'o'),
                    ms=6, capsize=3, lw=1.6, color=S.MODEL_COLORS.get(n, S.AQUA), label=n)
    ax.set_xticks(sorted(h['Horizon_min'].unique()))
    ax.set_xticklabels([f'{m} phút' if m < 60 else f'{m // 60} giờ' for m in sorted(h['Horizon_min'].unique())])
    ax.set_xlabel('Tầm dự báo')
    ax.set_ylabel('RMSE trên tập test (mg/L)')
    ax.set_title('RMSE theo tầm dự báo (đặc trưng gồm lag/rolling của COD; ± std qua 5 seed)')
    ax.legend(frameon=False)
    S.save(fig, FIG("fig12_horizon_rmse.png"))


# ---------------------------------------------------------------- fig13
def fig13_workflow():
    steps = [
        ('Dữ liệu thô', '6.544 file .txt, trạm LA_KCN_ANHHON\n8 thông số, chu kỳ 5 phút'),
        ('Kiểm tra dữ liệu', 'Trùng lặp, gap ≠ 5 phút, status ≠ 00\nChia đoạn liên tục'),
        ('Chia train/test theo thời gian', '80% đầu: train · 20% sau: test\n(trước mọi phép fit)'),
        ('Tiền xử lý (fit trên train)', 'ffill trong đoạn → median train\nClip IQR 7 cảm biến (không clip COD)'),
        ('Tạo cặp (t, t+h)', 'Chỉ cặp cách đúng h·5 phút\nKhông nối qua gap; purge biên train'),
        ('Huấn luyện', 'Persistence · LinearRegression · 6 mô hình cây\nStacking (TimeSeriesSplit 5 fold, meta LR)'),
        ('Đánh giá trên test', 'R², Adj R², RMSE, MAE, MAPE · 5 seed\nDiebold–Mariano, Wilcoxon · thời gian'),
        ('Giải thích & triển khai', 'SHAP Exact (Stacking), TreeSHAP (XGBoost/CatBoost)\nỨng dụng Streamlit'),
    ]
    fig, ax = plt.subplots(figsize=(S.WIDTH, S.WIDTH * 1.25))
    ax.set_xlim(0, 1), ax.set_ylim(0, len(steps)), ax.axis('off')
    bh, gap = 0.80, 1.0
    for i, (title, body) in enumerate(steps):
        y = len(steps) - (i + 1) * gap + (gap - bh) / 2
        fc = '#eaf2fc' if i not in (5, 6) else '#fdeee7'
        ax.add_patch(FancyBboxPatch((0.08, y), 0.84, bh, boxstyle='round,pad=0.005,rounding_size=0.02',
                                    fc=fc, ec=S.INK2, lw=0.8))
        ax.text(0.5, y + bh * 0.76, f'{i + 1}. {title}', ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=S.INK)
        ax.text(0.5, y + bh * 0.34, body, ha='center', va='center', fontsize=8, color=S.INK2,
                linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y - (gap - bh) + 0.01), xytext=(0.5, y - 0.01),
                        arrowprops=dict(arrowstyle='-|>', color=S.INK2, lw=1))
    ax.set_title('Quy trình xử lý và mô hình hoá', fontsize=11, fontweight='bold')
    S.save(fig, FIG("fig13_workflow.png"))


def main():
    ds = load_dataset(h=1)
    pr = pd.read_csv(os.path.join(C.TAB_DIR, "predictions_test_h1.csv"),
                     parse_dates=['Timestamp', 'Timestamp_target'])
    best = best_single_model()
    S.MODEL_COLORS[best] = S.AQUA   # mô hình đơn tốt nhất luôn dùng màu thứ 3
    fig01_correlation(ds)
    fig02_timeseries(ds)
    fig03_model_comparison()
    fig04_actual_vs_pred(pr)
    fig05_scatter(pr, best)
    fig07_lag_crosscorr(pr, best)
    fig12_horizon()
    fig13_workflow()


if __name__ == "__main__":
    main()
