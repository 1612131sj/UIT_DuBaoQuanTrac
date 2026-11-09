"""
Phân tích phần dư của Stacking (seed 42) trên tập test.
fig06_residuals.png: histogram, residuals vs fitted (+LOWESS), Q-Q plot.
tables/residual_stats.csv: mean, std, skew, kurtosis, Shapiro–Wilk, Jarque–Bera,
Breusch–Pagan, Ljung–Box (tự tương quan phần dư).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox

try:
    from . import config as C
    from . import plot_style as S
except ImportError:
    import config as C
    import plot_style as S

S.setup()


def residual_stats(name, y, yhat, X):
    e = y - yhat
    sw = stats.shapiro(e)
    jb = stats.jarque_bera(e)
    bp_fit = het_breuschpagan(e, sm.add_constant(yhat))
    bp_x = het_breuschpagan(e, sm.add_constant(X))
    lb = acorr_ljungbox(e, lags=[1, 6, 12], return_df=True)
    return {'Mô hình': name, 'n': len(e), 'Mean': e.mean(), 'Std': e.std(ddof=1),
            'Skewness': stats.skew(e), 'Kurtosis (excess)': stats.kurtosis(e),
            'Shapiro_W': sw.statistic, 'Shapiro_p': sw.pvalue,
            'JarqueBera_stat': jb.statistic, 'JarqueBera_p': jb.pvalue,
            'BP_LM (theo ŷ)': bp_fit[0], 'BP_p (theo ŷ)': bp_fit[1],
            'BP_LM (theo 8 đặc trưng)': bp_x[0], 'BP_p (theo 8 đặc trưng)': bp_x[1],
            'LjungBox_Q lag1': lb.loc[1, 'lb_stat'], 'LjungBox_p lag1': lb.loc[1, 'lb_pvalue'],
            'LjungBox_Q lag12': lb.loc[12, 'lb_stat'], 'LjungBox_p lag12': lb.loc[12, 'lb_pvalue'],
            'ACF lag1': sm.tsa.acf(e, nlags=1)[1]}


def main():
    pr = pd.read_csv(os.path.join(C.TAB_DIR, "predictions_test_h1.csv"))
    try:
        from .data_preprocessing import load_dataset
    except ImportError:
        from data_preprocessing import load_dataset
    X = load_dataset(h=1)['X_test']
    y = pr['y_true'].to_numpy()
    rows = [residual_stats(n, y, pr[n].to_numpy(), X)
            for n in ['Stacking', 'XGBoost', 'CatBoost', 'ExtraTrees', 'LinearRegression', 'Persistence']]
    pd.DataFrame(rows).to_csv(os.path.join(C.TAB_DIR, "residual_stats.csv"), index=False,
                              encoding='utf-8-sig')

    yhat = pr['Stacking'].to_numpy()
    e = y - yhat
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, S.WIDTH * 0.36))
    # (a) histogram
    axes[0].hist(e, bins=50, color=S.BLUE, alpha=0.85, edgecolor='white', linewidth=0.3)
    axes[0].axvline(0, color=S.INK, lw=0.9, ls='--')
    axes[0].set_xlabel('Phần dư = thực tế − dự báo (mg/L)')
    axes[0].set_ylabel('Số mẫu')
    axes[0].set_title('(a) Phân phối phần dư')
    # (b) residuals vs fitted + LOWESS
    axes[1].scatter(yhat, e, s=5, alpha=0.4, color=S.BLUE, edgecolors='none')
    lw = sm.nonparametric.lowess(e, yhat, frac=0.3)
    axes[1].plot(lw[:, 0], lw[:, 1], color=S.RED, lw=1.6, label='LOWESS')
    axes[1].axhline(0, color=S.INK, lw=0.9, ls='--')
    axes[1].set_xlabel('Giá trị dự báo (mg/L)')
    axes[1].set_ylabel('Phần dư (mg/L)')
    axes[1].set_title('(b) Phần dư – giá trị dự báo')
    axes[1].legend(frameon=False, loc='upper right')
    # (c) Q-Q plot
    (osm, osr), (slope, inter, _) = stats.probplot(e, dist='norm')
    axes[2].scatter(osm, osr, s=5, alpha=0.5, color=S.BLUE, edgecolors='none')
    axes[2].plot(osm, slope * osm + inter, color=S.INK, lw=0.9, ls='--')
    axes[2].set_xlabel('Phân vị lý thuyết N(0,1)')
    axes[2].set_ylabel('Phân vị phần dư (mg/L)', labelpad=2)
    axes[2].set_title('(c) Biểu đồ Q-Q')
    fig.suptitle(f'Phân tích phần dư mô hình Stacking trên tập test (n = {len(e)})', fontsize=10)
    fig.tight_layout(w_pad=2.5)
    S.save(fig, os.path.join(C.FIG_DIR, "fig06_residuals.png"))
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
