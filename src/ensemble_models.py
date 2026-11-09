"""
Định nghĩa mô hình: baseline Persistence, LinearRegression, 6 mô hình cơ sở và Stacking theo thời gian.

sklearn.ensemble.StackingRegressor không nhận cv=TimeSeriesSplit (cross_val_predict yêu cầu
các fold phủ kín toàn bộ dữ liệu, còn fold đầu của TimeSeriesSplit không bao giờ là tập kiểm định),
nên TimeSeriesStackingRegressor được cài đặt riêng:
  1) Với mỗi fold của TimeSeriesSplit: huấn luyện các mô hình cơ sở trên phần quá khứ,
     dự báo phần kế tiếp -> dự báo out-of-fold (OOF) không dùng dữ liệu tương lai.
  2) Huấn luyện meta-learner (LinearRegression) trên OOF.
  3) Huấn luyện lại các mô hình cơ sở trên toàn bộ train để dùng khi dự báo.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

try:
    from . import config as C
except ImportError:
    import config as C


class PersistenceRegressor(BaseEstimator, RegressorMixin):
    """Baseline ngây thơ: ŷ(t+h) = COD(t). Không học gì."""

    def __init__(self, col_idx: int = 0):
        self.col_idx = col_idx

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, self.col_idx]


class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking với meta-learner được học trên dự báo OOF từ TimeSeriesSplit."""

    def __init__(self, estimators, final_estimator=None, n_splits: int = 5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        tss = TimeSeriesSplit(n_splits=self.n_splits)
        oof = np.full((len(y), len(self.estimators)), np.nan)
        for tr, va in tss.split(X):
            for j, (_, est) in enumerate(self.estimators):
                oof[va, j] = clone(est).fit(X[tr], y[tr]).predict(X[va])
        mask = ~np.isnan(oof).any(axis=1)          # phần đầu (fold 0) không có OOF
        self.n_meta_samples_ = int(mask.sum())
        self.final_estimator_ = clone(self.final_estimator or LinearRegression())
        self.final_estimator_.fit(oof[mask], y[mask])
        # huấn luyện lại mô hình cơ sở trên toàn bộ train
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.named_estimators_ = {n: e for (n, _), e in zip(self.estimators, self.estimators_)}
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.column_stack([e.predict(X) for e in self.estimators_])

    def predict(self, X):
        return self.final_estimator_.predict(self.transform(X))


def base_models(seed: int = C.SEED):
    """6 mô hình cơ sở với siêu tham số mặc định (100 cây/vòng lặp)."""
    return {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=seed),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=seed),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=seed),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=seed),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=seed, objective='reg:squarederror'),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=seed, verbose=0,
                                      allow_writing_files=False),
    }


def with_scaler(model):
    """Bọc MinMaxScaler (fit trên train) + mô hình -> đầu vào luôn ở thang gốc (mg/L, ...)."""
    return Pipeline([('scaler', MinMaxScaler()), ('model', model)])


def build_models(seed: int = C.SEED, include=None):
    """Trả về dict tên -> mô hình (đều nhận đầu vào ở thang gốc)."""
    bases = base_models(seed)
    models = {'Persistence': PersistenceRegressor(col_idx=0),
              'LinearRegression': with_scaler(LinearRegression())}
    for n, m in bases.items():
        models[n] = with_scaler(m)
    models['Stacking'] = with_scaler(TimeSeriesStackingRegressor(
        estimators=list(base_models(seed).items()), final_estimator=LinearRegression(),
        n_splits=C.N_SPLITS_TS))
    if include is not None:
        models = {k: v for k, v in models.items() if k in include}
    return models


# Mô hình không phụ thuộc seed -> chỉ chạy 1 lần
DETERMINISTIC = {'Persistence', 'LinearRegression'}
