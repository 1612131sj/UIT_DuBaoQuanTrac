"""
Dự báo ở nhiều tầm h = 1, 6, 12, 24 bước (5 phút, 30 phút, 1 giờ, 2 giờ).
Đặc trưng = 8 thông số tại t + lag COD (t-1, t-2, t-3, t-6) + rolling mean/std (30 phút, 1 giờ)
+ hiệu COD(t) − COD(t−1). Mọi đặc trưng chỉ dùng quá khứ, tính trong từng đoạn liên tục.
So sánh Persistence / XGBoost / Stacking, 5 seed -> tables/horizon_metrics.csv.
"""
import os
import numpy as np
import pandas as pd

try:
    from . import config as C
    from .data_preprocessing import load_dataset
    from .ensemble_models import build_models
    from .evaluation import calculate_metrics, diebold_mariano
except ImportError:
    import config as C
    from data_preprocessing import load_dataset
    from ensemble_models import build_models
    from evaluation import calculate_metrics, diebold_mariano

MODELS = ['Persistence', 'XGBoost', 'Stacking']


def main():
    rows, sig = [], []
    for h in C.HORIZONS:
        ds = load_dataset(h=h, extended=True)
        Xtr, ytr, Xte, yte = ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']
        print(f"h={h}: train {Xtr.shape}, test {Xte.shape}")
        preds = {}
        for seed in C.SEEDS:
            inc = MODELS if seed == C.SEED else ['XGBoost', 'Stacking']
            for name, m in build_models(seed, include=inc).items():
                yp = m.fit(Xtr, ytr).predict(Xte)
                preds[(name, seed)] = yp
                met = calculate_metrics(yte, yp, n_features=1 if name == 'Persistence' else Xtr.shape[1])
                rows.append({'Horizon_steps': h, 'Horizon_min': h * C.STEP_MIN, 'Model': name,
                             'Seed': seed, 'n_train': len(ytr), 'n_test': len(yte), **met})
                print(f"   seed={seed} {name:<12} RMSE={met['RMSE']:.4f}")
        # DM: Stacking vs Persistence, XGBoost vs Persistence, Stacking vs XGBoost (seed 42)
        p = preds[('Persistence', C.SEED)]
        for a, b in [('Stacking', 'Persistence'), ('XGBoost', 'Persistence'), ('Stacking', 'XGBoost')]:
            ya = preds[(a, C.SEED)]
            yb = p if b == 'Persistence' else preds[(b, C.SEED)]
            st, pv, L, _ = diebold_mariano(yte - ya, yte - yb, h=h)
            sig.append({'Horizon_steps': h, 'So sánh': f'{a} vs {b}', 'DM_stat (MSE)': st,
                        'DM_p (MSE)': pv, 'HAC_lag': L})

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(C.TAB_DIR, "horizon_metrics_per_seed.csv"), index=False)
    g = per.groupby(['Horizon_steps', 'Horizon_min', 'Model'])
    out = g[['R2', 'RMSE', 'MAE', 'MAPE_%']].agg(['mean', 'std'])
    out.columns = [f'{a}_{b}' for a, b in out.columns]
    out = out.fillna(0.0)
    out['n_test'] = g['n_test'].first()
    out['n_seeds'] = g['Seed'].count()
    out = out.reset_index()
    pers = out[out['Model'] == 'Persistence'].set_index('Horizon_steps')['RMSE_mean']
    out['Skill_vs_Persistence'] = 1 - out['RMSE_mean'] / out['Horizon_steps'].map(pers)
    out['Model'] = pd.Categorical(out['Model'], MODELS)
    out = out.sort_values(['Horizon_steps', 'Model'])
    out.to_csv(os.path.join(C.TAB_DIR, "horizon_metrics.csv"), index=False)
    pd.DataFrame(sig).to_csv(os.path.join(C.TAB_DIR, "horizon_significance.csv"),
                             index=False, encoding='utf-8-sig')
    print(out[['Horizon_min', 'Model', 'R2_mean', 'RMSE_mean', 'RMSE_std', 'Skill_vs_Persistence']]
          .to_string(index=False))
    print(pd.DataFrame(sig).to_string(index=False))


if __name__ == "__main__":
    main()
