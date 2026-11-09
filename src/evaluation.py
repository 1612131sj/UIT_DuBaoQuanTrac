"""Chỉ số đánh giá và kiểm định thống kê so sánh dự báo."""
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred, n_features):
    """R², Adj R², RMSE, MAE, MAPE(%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {'R2': r2, 'Adj_R2': adj,
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'MAPE_%': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)}


def newey_west_lag(n: int) -> int:
    """Băng thông Newey–West tự động: floor(4·(n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def diebold_mariano(e1, e2, h: int = 1, power: int = 2, lag=None):
    """
    Kiểm định Diebold–Mariano (có hiệu chỉnh Harvey–Leybourne–Newbold).
    H0: hai dự báo có độ chính xác kỳ vọng bằng nhau. d_t = |e1|^p - |e2|^p.
    DM < 0 nghĩa là mô hình 1 có sai số nhỏ hơn mô hình 2.
    Phương sai dài hạn của d_t ước lượng bằng HAC (Newey–West, kernel Bartlett) với độ trễ
    max(h-1, newey_west_lag(n)) vì sai số dự báo chuỗi thời gian thường tự tương quan.
    """
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n = len(d)
    L = max(h - 1, newey_west_lag(n)) if lag is None else lag
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        lrv += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p), L, float(d.mean())


def wilcoxon_abs(e1, e2):
    """Wilcoxon signed-rank trên |e1| - |e2| (không giả định chuẩn; bỏ qua tự tương quan)."""
    res = stats.wilcoxon(np.abs(e1), np.abs(e2))
    return float(res.statistic), float(res.pvalue)
