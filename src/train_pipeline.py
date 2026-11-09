"""
Huấn luyện, so sánh với baseline, kiểm định ý nghĩa thống kê và đo thời gian.

Đầu ra (outputs/v2/tables):
  model_metrics.csv           mean ± std qua 5 seed (42–46) + Skill so với Persistence
  model_metrics_per_seed.csv  số liệu từng seed
  significance.csv            Diebold–Mariano + Wilcoxon
  meta_weights.csv            hệ số meta-learner
  timing.csv                  thời gian huấn luyện / suy luận
  predictions_test_h1.csv     dự báo trên test (seed 42) – dùng để vẽ hình
  sensitivity_clip_cod.csv    độ nhạy khi clip IQR cả COD (seed 42)
  stacking_variants.csv       các biến thể Stacking (seed 42)
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd

try:
    from . import config as C
    from .data_preprocessing import load_dataset
    from .ensemble_models import build_models, DETERMINISTIC
    from .evaluation import calculate_metrics, diebold_mariano, wilcoxon_abs
except ImportError:
    import config as C
    from data_preprocessing import load_dataset
    from ensemble_models import build_models, DETERMINISTIC
    from evaluation import calculate_metrics, diebold_mariano, wilcoxon_abs

MODEL_ORDER = ['Persistence', 'LinearRegression', 'RandomForest', 'ExtraTrees',
               'GradientBoosting', 'AdaBoost', 'XGBoost', 'CatBoost', 'Stacking']


def time_inference(model, X, repeats=C.N_TIMING_REPEATS):
    """Median thời gian dự báo toàn bộ X qua `repeats` lần -> ms / mẫu."""
    model.predict(X[:10])  # khởi động (warm-up)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(X) * 1000, float(np.median(ts))


def run_one_seed(ds, seed, include=None, save=False, do_timing=False):
    """Huấn luyện mọi mô hình với 1 seed. Trả về (metrics list, predictions dict, timing list, models)."""
    Xtr, ytr, Xte, yte = ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']
    models = build_models(seed, include=include)
    rows, preds, timing, fitted = [], {}, [], {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        t_train = time.perf_counter() - t0
        yp = m.predict(Xte)
        preds[name] = yp
        met = calculate_metrics(yte, yp, n_features=1 if name == 'Persistence' else Xtr.shape[1])
        rows.append({'Model': name, 'Seed': seed, **met, 'Train_time_s': t_train})
        if do_timing:
            ms, tot = time_inference(m, Xte)
            timing.append({'Model': name, 'Seed': seed, 'Train_time_s': t_train,
                           'Infer_total_s_median': tot, 'Infer_ms_per_sample_median': ms})
        fitted[name] = m
        if save:
            joblib.dump(m, os.path.join(C.MODELS_DIR, f"{name}.pkl"))
        print(f"   seed={seed} {name:<17} RMSE={met['RMSE']:.4f}  train={t_train:.2f}s")
    return rows, preds, timing, fitted


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """mean ± std qua seed; thêm Skill = 1 − RMSE_model / RMSE_persistence."""
    agg = per_seed.groupby('Model').agg(['mean', 'std', 'count'])
    out = pd.DataFrame(index=agg.index)
    for k in ['R2', 'Adj_R2', 'RMSE', 'MAE', 'MAPE_%']:
        out[f'{k}_mean'] = agg[(k, 'mean')]
        out[f'{k}_std'] = agg[(k, 'std')].fillna(0.0)
    out['n_seeds'] = agg[('R2', 'count')]
    rmse_p = out.loc['Persistence', 'RMSE_mean']
    # skill tính theo từng seed rồi lấy mean ± std
    sk = per_seed.assign(Skill=1 - per_seed['RMSE'] / rmse_p).groupby('Model')['Skill']
    out['Skill_vs_Persistence_mean'] = sk.mean()
    out['Skill_vs_Persistence_std'] = sk.std().fillna(0.0)
    out = out.reindex([m for m in MODEL_ORDER if m in out.index])
    return out.reset_index()


def main():
    ds = load_dataset(h=1, extended=False)
    Xtr, ytr, Xte, yte = ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    # Lưu khoảng min–max của train (cho app) và mẫu nền SHAP
    tr_df = pd.DataFrame(Xtr, columns=ds['features'])
    ranges = {c: [float(tr_df[c].min()), float(tr_df[c].max()), float(tr_df[c].median())]
              for c in ds['features']}
    with open(os.path.join(C.MODELS_DIR, "train_ranges.json"), 'w', encoding='utf-8') as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)
    rng = np.random.default_rng(C.SEED)
    bg_idx = np.sort(rng.choice(len(Xtr), C.SHAP_BACKGROUND_N, replace=False))
    np.save(os.path.join(C.MODELS_DIR, "shap_background.npy"), Xtr[bg_idx])

    all_rows, timing, preds42, fitted42 = [], [], None, None
    for seed in C.SEEDS:
        include = None if seed == C.SEEDS[0] else [m for m in MODEL_ORDER if m not in DETERMINISTIC]
        rows, preds, tim, fitted = run_one_seed(ds, seed, include=include,
                                                save=(seed == C.SEED), do_timing=True)
        all_rows += rows
        timing += tim
        if seed == C.SEED:
            preds42, fitted42 = preds, fitted
        else:
            preds42.update({f'{k}__seed{seed}': v for k, v in preds.items()})

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(os.path.join(C.TAB_DIR, "model_metrics_per_seed.csv"), index=False)
    summary = summarize(per_seed)
    summary.to_csv(os.path.join(C.TAB_DIR, "model_metrics.csv"), index=False)
    print("\n", summary[['Model', 'R2_mean', 'RMSE_mean', 'RMSE_std', 'MAE_mean',
                         'MAPE_%_mean', 'Skill_vs_Persistence_mean']].to_string(index=False))

    # ---- dự báo test (seed 42 + các seed khác) để vẽ hình
    pr = ds['test_df'][['Timestamp', 'Timestamp_target', 'segment']].copy()
    pr['y_true'] = yte
    for k, v in preds42.items():
        pr[k] = v
    pr.to_csv(os.path.join(C.TAB_DIR, "predictions_test_h1.csv"), index=False)

    # ---- kiểm định ý nghĩa
    singles = summary[~summary['Model'].isin(['Persistence', 'Stacking'])]
    best_single = singles.sort_values('RMSE_mean').iloc[0]['Model']
    print(f"\nMô hình đơn tốt nhất (RMSE trung bình): {best_single}")
    sig = []
    pairs = [('Stacking', best_single), ('Stacking', 'Persistence'), (best_single, 'Persistence')]
    for seed in C.SEEDS:
        def get(name):
            if name in DETERMINISTIC or seed == C.SEED:
                return pr[name].to_numpy()
            return pr[f'{name}__seed{seed}'].to_numpy()
        for a, b in pairs:
            ea, eb = yte - get(a), yte - get(b)
            dm2, p2, L, dmean = diebold_mariano(ea, eb, h=1, power=2)
            dm1, p1, _, _ = diebold_mariano(ea, eb, h=1, power=1)
            w, pw = wilcoxon_abs(ea, eb)
            sig.append({'So sánh (A vs B)': f'{a} vs {b}', 'Seed': seed,
                        'RMSE_A': float(np.sqrt(np.mean(ea ** 2))), 'RMSE_B': float(np.sqrt(np.mean(eb ** 2))),
                        'Mean(e_A² − e_B²)': dmean,
                        'DM_stat (MSE)': dm2, 'DM_p (MSE)': p2,
                        'DM_stat (MAE)': dm1, 'DM_p (MAE)': p1, 'HAC_lag': L,
                        'Wilcoxon_stat |e|': w, 'Wilcoxon_p |e|': pw,
                        'Median(|e_A| − |e_B|)': float(np.median(np.abs(ea) - np.abs(eb)))})
    sig = pd.DataFrame(sig)
    sig.to_csv(os.path.join(C.TAB_DIR, "significance.csv"), index=False, encoding='utf-8-sig')
    print(sig[['So sánh (A vs B)', 'Seed', 'RMSE_A', 'RMSE_B', 'DM_stat (MSE)', 'DM_p (MSE)',
               'Wilcoxon_p |e|']].to_string(index=False))

    # ---- trọng số meta-learner (mọi seed)
    mw = []
    for seed in C.SEEDS:
        m = fitted42['Stacking'] if seed == C.SEED else None
        if m is None:
            # huấn luyện lại stacking cho seed này chỉ để lấy hệ số (không tính vào metrics)
            m = build_models(seed, include=['Stacking'])['Stacking'].fit(Xtr, ytr)
        st = m.named_steps['model']
        row = {'Seed': seed, 'Intercept': float(st.final_estimator_.intercept_),
               'n_meta_samples': st.n_meta_samples_}
        for (n, _), c in zip(st.estimators, st.final_estimator_.coef_):
            row[n] = float(c)
        row['Tổng hệ số'] = float(np.sum(st.final_estimator_.coef_))
        mw.append(row)
    mw = pd.DataFrame(mw)
    mw.to_csv(os.path.join(C.TAB_DIR, "meta_weights.csv"), index=False, encoding='utf-8-sig')
    print("\nTrọng số meta-learner:\n", mw.round(4).to_string(index=False))

    # ---- thời gian
    tim = pd.DataFrame(timing)
    tim.to_csv(os.path.join(C.TAB_DIR, "timing_per_seed.csv"), index=False)
    tsum = tim.groupby('Model').agg(
        Train_time_s_median=('Train_time_s', 'median'), Train_time_s_min=('Train_time_s', 'min'),
        Train_time_s_max=('Train_time_s', 'max'),
        Infer_ms_per_sample_median=('Infer_ms_per_sample_median', 'median'),
        Infer_total_s_median=('Infer_total_s_median', 'median'), n_runs=('Seed', 'count'))
    tsum = tsum.reindex([m for m in MODEL_ORDER if m in tsum.index]).reset_index()
    tsum['n_test'] = len(Xte)
    tsum.to_csv(os.path.join(C.TAB_DIR, "timing.csv"), index=False)
    print("\nThời gian:\n", tsum.to_string(index=False))

    # ---- độ nhạy: clip IQR cả COD (seed 42), đánh giá trên COD thô
    ds_c = load_dataset(h=1, extended=False, clip_cod=True)
    rows_c, _, _, _ = run_one_seed(ds_c, C.SEED)
    sc = pd.DataFrame(rows_c)[['Model', 'R2', 'RMSE', 'MAE', 'MAPE_%']]
    base = per_seed[per_seed['Seed'] == C.SEED].set_index('Model')
    sc['RMSE (không clip COD, seed 42)'] = sc['Model'].map(base['RMSE'])
    sc.to_csv(os.path.join(C.TAB_DIR, "sensitivity_clip_cod.csv"), index=False)
    print("\nĐộ nhạy clip COD:\n", sc.to_string(index=False))

    stacking_variants(ds)


def stacking_variants(ds):
    """
    Phân tích bổ sung (seed 42, KHÔNG thay mô hình chính): Stacking thay đổi cách tạo OOF / meta-learner.
      - TSS + LR             : mô hình chính
      - KFold(5) + LR        : StackingRegressor của sklearn với cv=5 – OOF dùng cả dữ liệu tương lai
      - TSS + LR(positive)   : meta ràng buộc hệ số ≥ 0
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    try:
        from .ensemble_models import base_models, with_scaler, TimeSeriesStackingRegressor
    except ImportError:
        from ensemble_models import base_models, with_scaler, TimeSeriesStackingRegressor
    Xtr, ytr, Xte, yte = ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']
    variants = {
        'TSS(5) + LinearRegression (mô hình chính)': with_scaler(TimeSeriesStackingRegressor(
            list(base_models(C.SEED).items()), LinearRegression(), C.N_SPLITS_TS)),
        'KFold(5) + LinearRegression (phiên bản trước)': with_scaler(StackingRegressor(
            list(base_models(C.SEED).items()), final_estimator=LinearRegression(), cv=KFold(5))),
        'TSS(5) + LinearRegression(positive=True)': with_scaler(TimeSeriesStackingRegressor(
            list(base_models(C.SEED).items()), LinearRegression(positive=True), C.N_SPLITS_TS)),
    }
    rows = []
    for name, m in variants.items():
        m.fit(Xtr, ytr)
        met = calculate_metrics(yte, m.predict(Xte), n_features=Xtr.shape[1])
        fe = m.named_steps['model'].final_estimator_
        rows.append({'Biến thể': name, **met, 'Intercept meta': float(fe.intercept_),
                     'Tổng hệ số meta': float(np.sum(fe.coef_)),
                     **{f'coef_{n}': float(c) for (n, _), c in zip(base_models(C.SEED).items(), fe.coef_)}})
        print(f"   {name}: RMSE={met['RMSE']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(C.TAB_DIR, "stacking_variants.csv"), index=False,
                              encoding='utf-8-sig')


if __name__ == "__main__":
    main()
