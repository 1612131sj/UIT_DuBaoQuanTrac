"""
Đọc dữ liệu thô, kiểm tra chất lượng dữ liệu và tiền xử lý không rò rỉ thông tin từ tập test.

- Đọc cả cột trạng thái (status). Giá trị có status khác '00' bị coi là không hợp lệ -> NaN.
- Kiểm tra Timestamp: trùng lặp, khoảng trống (gap != 5 phút), thời gian bao phủ.
- Chia chuỗi thành các "đoạn liên tục" (segment); không nội suy / không tạo cặp (t, t+h) qua gap.
- Điền thiếu: forward-fill trong cùng đoạn (chỉ dùng quá khứ) -> nếu còn thiếu thì median của TRAIN.
- Ngưỡng IQR (clip) chỉ fit trên TRAIN (80% đầu theo thời gian), rồi áp dụng cho test.
"""
import os
import glob
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:  # chạy trực tiếp: python src/data_preprocessing.py
    import config as C

RAW_CACHE = os.path.join(C.PROCESSED_DIR, "water_quality_raw_v2.csv")


# =====================================================================
# 1. Đọc dữ liệu thô
# =====================================================================
def parse_txt_file(filepath: str) -> dict:
    """Đọc 1 file .txt (tab-separated: Thông số, Giá trị, Đơn vị, Thời điểm, Trạng thái)."""
    name = os.path.basename(filepath)[:-4]
    parts = name.split('_')
    rec = {'Station': '_'.join(parts[:-1]), 'Timestamp': parts[-1]}
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            cols = line.rstrip('\r\n').split('\t')
            if len(cols) < 2:
                continue
            p = cols[0].strip()
            try:
                rec[p] = float(cols[1])
            except ValueError:
                rec[p] = np.nan
            rec[p + '_status'] = cols[4].strip() if len(cols) >= 5 else ''
    return rec


def load_raw(raw_dir: str = C.RAW_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Gộp toàn bộ file thô thành DataFrame (có cache để chạy lại nhanh)."""
    if use_cache and os.path.exists(RAW_CACHE):
        df = pd.read_csv(RAW_CACHE, parse_dates=['Timestamp'],
                         dtype={f + '_status': str for f in C.FEATURES})
        return df
    files = glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .txt trong {raw_dir}")
    df = pd.DataFrame([parse_txt_file(f) for f in files])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S', errors='coerce')
    df = df.sort_values(['Station', 'Timestamp']).reset_index(drop=True)
    df.to_csv(RAW_CACHE, index=False)
    return df


# =====================================================================
# 2. Kiểm tra thời gian & chia đoạn liên tục
# =====================================================================
def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Gán mã đoạn liên tục: đoạn mới bắt đầu mỗi khi khoảng cách != 5 phút."""
    df = df.copy()
    gap = df['Timestamp'].diff().dt.total_seconds() / 60
    df['segment'] = (gap != C.STEP_MIN).cumsum() - 1   # dòng đầu tiên: gap = NaN -> đoạn 0
    return df


def train_cutoff(df: pd.DataFrame) -> pd.Timestamp:
    """Mốc chia train/test: bản ghi thứ floor(80%·N) là bản ghi test đầu tiên."""
    return df['Timestamp'].iloc[int(len(df) * C.TRAIN_FRAC)]


def mask_invalid_status(df: pd.DataFrame):
    """Giá trị có status không hợp lệ -> NaN. Trả về df và số giá trị bị loại theo cột."""
    df = df.copy()
    counts = {}
    for c in C.FEATURES:
        sc = c + '_status'
        if sc in df:
            bad = df[sc].notna() & ~df[sc].astype(str).str.zfill(2).isin(C.VALID_STATUS)
            counts[c] = int(bad.sum())
            df.loc[bad, c] = np.nan
        else:
            counts[c] = 0
    return df, counts


# =====================================================================
# 3. Tiền xử lý: fit trên train, transform cho cả train và test
# =====================================================================
class Preprocessor:
    """Điền thiếu + clip IQR, tham số chỉ học từ tập train."""

    def __init__(self, clip_cod: bool = C.CLIP_COD, k: float = C.IQR_K):
        self.clip_cod = clip_cod
        self.k = k

    def fit(self, train_df: pd.DataFrame):
        self.medians_ = train_df[C.FEATURES].median()
        q1 = train_df[C.FEATURES].quantile(0.25)
        q3 = train_df[C.FEATURES].quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.clip_cols_ = [c for c in C.FEATURES if self.clip_cod or c != C.TARGET]
        return self

    def transform(self, df: pd.DataFrame):
        """Trả về (df đã xử lý, bảng đếm số giá trị được điền / bị clip)."""
        out = df.copy()
        log = pd.DataFrame(0, index=C.FEATURES,
                           columns=['n_ffill', 'n_median_fill', 'n_clip_low', 'n_clip_high'])
        for c in C.FEATURES:
            na0 = out[c].isna()
            # forward-fill trong cùng một đoạn liên tục (chỉ dùng quá khứ, không nối qua gap)
            out[c] = out.groupby('segment')[c].ffill()
            na1 = out[c].isna()
            log.loc[c, 'n_ffill'] = int((na0 & ~na1).sum())
            # nếu đầu đoạn vẫn thiếu -> median của train
            out[c] = out[c].fillna(self.medians_[c])
            log.loc[c, 'n_median_fill'] = int(na1.sum())
            if c in self.clip_cols_:
                lo, hi = self.lower_[c], self.upper_[c]
                log.loc[c, 'n_clip_low'] = int((out[c] < lo).sum())
                log.loc[c, 'n_clip_high'] = int((out[c] > hi).sum())
                out[c] = out[c].clip(lo, hi)
        return out, log


def preprocess(df_raw: pd.DataFrame, clip_cod: bool = C.CLIP_COD):
    """
    Toàn bộ chuỗi tiền xử lý. Trả về dict gồm:
      clean   : DataFrame đã xử lý (cột 'is_train' đánh dấu giai đoạn train)
      cutoff  : mốc thời gian chia train/test
      prep    : đối tượng Preprocessor đã fit
      log_train, log_test, status_counts
    """
    df = df_raw.drop_duplicates(subset='Timestamp', keep='first').reset_index(drop=True)
    df = add_segments(df)
    df, status_counts = mask_invalid_status(df)
    cutoff = train_cutoff(df)
    is_train = df['Timestamp'] < cutoff
    prep = Preprocessor(clip_cod=clip_cod).fit(df[is_train])
    tr, log_tr = prep.transform(df[is_train])
    te, log_te = prep.transform(df[~is_train])
    clean = pd.concat([tr, te]).sort_values('Timestamp').reset_index(drop=True)
    clean['is_train'] = clean['Timestamp'] < cutoff
    # giữ lại COD thô (chưa clip) để làm "đáp án" khi đánh giá
    clean['COD_raw'] = df['COD'].values
    return dict(clean=clean, cutoff=cutoff, prep=prep, log_train=log_tr,
                log_test=log_te, status_counts=status_counts)


# =====================================================================
# 4. Tạo bài toán có giám sát (t -> t+h), không nối qua gap
# =====================================================================
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng trễ / trượt của COD, chỉ dùng dữ liệu quá khứ, tính trong từng đoạn liên tục."""
    df = df.copy()
    g = df.groupby('segment')['COD']
    for L in (1, 2, 3, 6):
        df[f'COD_lag{L}'] = g.shift(L)
    for w, name in ((6, '30m'), (12, '1h')):          # 6 bước = 30 phút, 12 bước = 1 giờ
        df[f'COD_rmean_{name}'] = g.transform(lambda s: s.rolling(w, min_periods=w).mean())
        df[f'COD_rstd_{name}'] = g.transform(lambda s: s.rolling(w, min_periods=w).std())
    df['COD_diff1'] = df['COD'] - df['COD_lag1']
    return df


LAG_FEATURES = ['COD_lag1', 'COD_lag2', 'COD_lag3', 'COD_lag6',
                'COD_rmean_30m', 'COD_rstd_30m', 'COD_rmean_1h', 'COD_rstd_1h', 'COD_diff1']


def make_supervised(clean: pd.DataFrame, cutoff, h: int = 1, extended: bool = False):
    """
    Tạo (X, y) với y = COD_raw(t+h). Chỉ giữ cặp mà t và t+h cùng đoạn liên tục
    (tức cách nhau đúng h·5 phút). Cặp train có t+h rơi vào giai đoạn test bị loại (purge).
    """
    df = add_lag_features(clean) if extended else clean.copy()
    g = df.groupby('segment')
    df['y_target'] = g['COD_raw'].shift(-h)
    df['Timestamp_target'] = g['Timestamp'].shift(-h)
    feats = C.FEATURES + (LAG_FEATURES if extended else [])
    df = df.dropna(subset=feats + ['y_target'])
    # kiểm tra lại khoảng cách đúng h·5 phút
    dt = (df['Timestamp_target'] - df['Timestamp']).dt.total_seconds() / 60
    assert (dt == h * C.STEP_MIN).all()
    train = df[df['Timestamp_target'] < cutoff]
    test = df[df['Timestamp'] >= cutoff]
    return dict(X_train=train[feats].to_numpy(float), y_train=train['y_target'].to_numpy(float),
                X_test=test[feats].to_numpy(float), y_test=test['y_target'].to_numpy(float),
                features=feats, train_df=train, test_df=test)


def load_dataset(h: int = 1, extended: bool = False, clip_cod: bool = C.CLIP_COD):
    """Hàm tiện ích cho các script khác: đọc thô -> tiền xử lý -> (X, y)."""
    res = preprocess(load_raw(), clip_cod=clip_cod)
    ds = make_supervised(res['clean'], res['cutoff'], h=h, extended=extended)
    ds.update(res)
    return ds


# =====================================================================
# 5. Kiểm tra dữ liệu + thống kê mô tả
# =====================================================================
def run_audit_and_stats():
    df_raw = load_raw(use_cache=False)
    n = len(df_raw)
    ts = df_raw['Timestamp']
    n_dup = int(ts.duplicated().sum())
    d = df_raw.drop_duplicates(subset='Timestamp').reset_index(drop=True)
    d = add_segments(d)
    gaps_min = d['Timestamp'].diff().dt.total_seconds() / 60
    gap_rows = d[gaps_min.notna() & (gaps_min != C.STEP_MIN)]
    span = d['Timestamp'].iloc[-1] - d['Timestamp'].iloc[0]
    expected = int(span.total_seconds() / 60 / C.STEP_MIN) + 1

    # --- bảng các khoảng trống
    gaps = pd.DataFrame({
        'Từ (bản ghi trước gap)': d['Timestamp'].shift(1)[gap_rows.index].values,
        'Đến (bản ghi sau gap)': gap_rows['Timestamp'].values,
        'Khoảng cách (phút)': gaps_min[gap_rows.index].values,
        'Số chu kỳ bị thiếu': (gaps_min[gap_rows.index].values / C.STEP_MIN - 1).astype(int),
    })
    gaps.to_csv(os.path.join(C.TAB_DIR, "data_gaps.csv"), index=False, encoding='utf-8-sig')

    res = preprocess(df_raw)
    clean, cutoff = res['clean'], res['cutoff']
    sup = make_supervised(clean, cutoff, h=1)
    n_pairs_naive = len(d) - 1
    n_pairs_valid = int((gaps_min.shift(-1) == C.STEP_MIN).sum())

    # --- bảng tóm tắt thời gian
    time_rows = [
        ('Số bản ghi (file)', n),
        ('Số Timestamp trùng lặp', n_dup),
        ('Bắt đầu', d['Timestamp'].iloc[0]),
        ('Kết thúc', d['Timestamp'].iloc[-1]),
        ('Tổng thời gian bao phủ (ngày)', round(span.total_seconds() / 86400, 4)),
        ('Số chu kỳ 5 phút kỳ vọng', expected),
        ('Số chu kỳ bị thiếu', expected - len(d)),
        ('Tỉ lệ đầy đủ theo thời gian (%)', round(100 * len(d) / expected, 3)),
        ('Số khoảng trống (gap != 5 phút)', len(gap_rows)),
        ('Số đoạn liên tục', int(d['segment'].nunique())),
        ('Số cặp (t, t+1) nếu shift thẳng (cách cũ)', n_pairs_naive),
        ('Số cặp (t, t+1) hợp lệ (cách đúng 5 phút)', n_pairs_valid),
        ('Số cặp bị loại vì nối qua gap', n_pairs_naive - n_pairs_valid),
        ('Mốc chia train/test (bản ghi test đầu tiên)', cutoff),
        ('Số bản ghi giai đoạn train', int(clean['is_train'].sum())),
        ('Số bản ghi giai đoạn test', int((~clean['is_train']).sum())),
        ('Số cặp train (h=1, sau purge)', len(sup['y_train'])),
        ('Số cặp test (h=1)', len(sup['y_test'])),
    ]
    pd.DataFrame(time_rows, columns=['Mục', 'Giá trị']).to_csv(
        os.path.join(C.TAB_DIR, "data_audit_time.csv"), index=False, encoding='utf-8-sig')

    # --- bảng audit theo cột
    prep, ltr, lte = res['prep'], res['log_train'], res['log_test']
    rows = []
    for c in C.FEATURES:
        x = d[c]
        rows.append({
            'Cột': c, 'Đơn vị': C.UNITS[c],
            'Số giá trị': int(x.notna().sum()),
            'Thiếu (NaN khi đọc)': int(x.isna().sum()),
            'Status ≠ 00 (coi là thiếu)': res['status_counts'][c],
            'Tỉ lệ thiếu (%)': round(100 * (x.isna().sum() + res['status_counts'][c]) / len(d), 4),
            'Điền bằng ffill': int(ltr.loc[c, 'n_ffill'] + lte.loc[c, 'n_ffill']),
            'Điền bằng median train': int(ltr.loc[c, 'n_median_fill'] + lte.loc[c, 'n_median_fill']),
            'Min thô': x.min(), 'Max thô': x.max(),
            'Ngưỡng IQR dưới (train)': prep.lower_[c], 'Ngưỡng IQR trên (train)': prep.upper_[c],
            'Có clip?': 'Có' if c in prep.clip_cols_ else 'Không',
            'Clip dưới (train)': int(ltr.loc[c, 'n_clip_low']),
            'Clip trên (train)': int(ltr.loc[c, 'n_clip_high']),
            'Clip dưới (test)': int(lte.loc[c, 'n_clip_low']),
            'Clip trên (test)': int(lte.loc[c, 'n_clip_high']),
            # số giá trị nằm ngoài ngưỡng (kể cả cột không clip, ví dụ COD) để tham khảo
            'Ngoài ngưỡng IQR (train)': int(((d.loc[d['Timestamp'] < cutoff, c] < prep.lower_[c]) |
                                             (d.loc[d['Timestamp'] < cutoff, c] > prep.upper_[c])).sum()),
            'Ngoài ngưỡng IQR (test)': int(((d.loc[d['Timestamp'] >= cutoff, c] < prep.lower_[c]) |
                                            (d.loc[d['Timestamp'] >= cutoff, c] > prep.upper_[c])).sum()),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(os.path.join(C.TAB_DIR, "data_audit.csv"), index=False, encoding='utf-8-sig')

    # --- thống kê mô tả SAU tiền xử lý (toàn giai đoạn + tách train/test cho COD)
    desc = []
    for c in C.FEATURES:
        x = clean[c]
        desc.append({'Thông số': c, 'Đơn vị': C.UNITS[c], 'N': len(x), 'Mean': x.mean(), 'Std': x.std(),
                     'Min': x.min(), 'Max': x.max(), 'Q1': x.quantile(.25), 'Q3': x.quantile(.75),
                     'Skewness': x.skew(),
                     'Số giá trị bị clip': int(ltr.loc[c, ['n_clip_low', 'n_clip_high']].sum() +
                                               lte.loc[c, ['n_clip_low', 'n_clip_high']].sum())})
    pd.DataFrame(desc).to_csv(os.path.join(C.TAB_DIR, "descriptive_stats.csv"),
                              index=False, encoding='utf-8-sig')
    # COD theo giai đoạn (để thấy dịch chuyển phân phối train/test)
    split_desc = clean.groupby(np.where(clean['is_train'], 'Train', 'Test'))['COD'].describe()
    split_desc['skew'] = clean.groupby(np.where(clean['is_train'], 'Train', 'Test'))['COD'].skew()
    split_desc.to_csv(os.path.join(C.TAB_DIR, "cod_train_vs_test.csv"), encoding='utf-8-sig')

    clean.to_csv(os.path.join(C.PROCESSED_DIR, "water_quality_cleaned_v2.csv"), index=False)

    # --- in tóm tắt
    print("=== AUDIT THỜI GIAN ===")
    for k, v in time_rows:
        print(f"  {k}: {v}")
    print("\n=== CÁC KHOẢNG TRỐNG ===\n", gaps.to_string(index=False))
    print("\n=== AUDIT THEO CỘT ===")
    print(audit[['Cột', 'Tỉ lệ thiếu (%)', 'Status ≠ 00 (coi là thiếu)', 'Có clip?', 'Clip dưới (train)',
                 'Clip trên (train)', 'Clip dưới (test)', 'Clip trên (test)',
                 'Ngoài ngưỡng IQR (train)', 'Ngoài ngưỡng IQR (test)']].to_string(index=False))
    print("\n=== COD train vs test ===\n", split_desc.round(3).to_string())
    return audit


if __name__ == "__main__":
    run_audit_and_stats()
