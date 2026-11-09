# KẾT QUẢ PIPELINE v2 – DỰ BÁO COD TRẠM LA_KCN_ANHHON

*Tệp này được sinh tự động bởi `src/make_results.py` lúc 2026-11-09 08:57 từ các file trong `outputs/v2/tables/`. Không có con số nào được nhập tay.*

Chạy lại toàn bộ: `python run_all.py`. Số thập phân dùng dấu chấm; đơn vị COD là mg/L.

## 0. Môi trường chạy

```
Thời điểm ghi: 2026-11-09 07:48:19
OS: Windows-11-10.0.26200-SP0
CPU: 13th Gen Intel(R) Core(TM) i5-1335U
Số luồng logic (os.cpu_count): 12
RAM: 15.6 GB
Python: 3.13.9
Seed chính: 42; các seed: [42, 43, 44, 45, 46]
Ghi chú luồng: sklearn mặc định 1 luồng (n_jobs=None); XGBoost và CatBoost mặc định dùng mọi luồng.

Phiên bản thư viện:
  numpy==2.5.3
  pandas==3.0.6
  sklearn==1.9.1
  xgboost==3.4.1
  catboost==1.2.10
  shap==0.52.0
  scipy==1.18.1
  statsmodels==0.15.0
  matplotlib==3.11.2
  seaborn==0.13.2
  streamlit==1.64.0
  joblib==1.6.0
```

## 1. Kiểm tra dữ liệu

**Bảng 1. Tóm tắt kiểm tra trục thời gian** (`tables/data_audit_time.csv`)

| Mục | Giá trị |
|---|---|
| Số bản ghi (file) | 6544 |
| Số Timestamp trùng lặp | 0 |
| Bắt đầu | 2026-01-01 00:05:00 |
| Kết thúc | 2026-01-23 21:50:00 |
| Tổng thời gian bao phủ (ngày) | 22.9062 |
| Số chu kỳ 5 phút kỳ vọng | 6598 |
| Số chu kỳ bị thiếu | 54 |
| Tỉ lệ đầy đủ theo thời gian (%) | 99.182 |
| Số khoảng trống (gap != 5 phút) | 4 |
| Số đoạn liên tục | 5 |
| Số cặp (t, t+1) nếu shift thẳng (cách cũ) | 6543 |
| Số cặp (t, t+1) hợp lệ (cách đúng 5 phút) | 6539 |
| Số cặp bị loại vì nối qua gap | 4 |
| Mốc chia train/test (bản ghi test đầu tiên) | 2026-01-19 05:15:00 |
| Số bản ghi giai đoạn train | 5235 |
| Số bản ghi giai đoạn test | 1309 |
| Số cặp train (h=1, sau purge) | 5231 |
| Số cặp test (h=1) | 1307 |

**Bảng 2. Các khoảng trống thời gian (gap ≠ 5 phút)** (`tables/data_gaps.csv`)

| Từ (bản ghi trước gap) | Đến (bản ghi sau gap) | Khoảng cách (phút) | Số chu kỳ bị thiếu |
|---|---|---|---|
| 2026-01-01 00:55:00 | 2026-01-01 01:05:00 | 10 | 1 |
| 2026-01-01 01:20:00 | 2026-01-01 02:00:00 | 40 | 7 |
| 2026-01-01 02:00:00 | 2026-01-01 02:20:00 | 20 | 3 |
| 2026-01-22 13:00:00 | 2026-01-22 16:40:00 | 220 | 43 |

**Bảng 3. Kiểm tra theo cột: thiếu, điền, ngưỡng IQR (fit trên train) và số giá trị bị clip** (`tables/data_audit.csv`)

| Cột | Đơn vị | Tỉ lệ thiếu (%) | Status ≠ 00 (coi là thiếu) | Điền bằng ffill | Điền bằng median train | Ngưỡng IQR dưới (train) | Ngưỡng IQR trên (train) | Có clip? | Clip dưới (train) | Clip trên (train) | Clip dưới (test) | Clip trên (test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COD | mg/L | 0.000 | 0 | 0 | 0 | 2.370 | 22.050 | Không | 0 | 0 | 0 | 0 |
| TSS | mg/L | 0.000 | 0 | 0 | 0 | 8.945 | 10.105 | Có | 393 | 226 | 0 | 101 |
| pH | - | 0.000 | 0 | 0 | 0 | 7.680 | 8.560 | Có | 172 | 87 | 33 | 55 |
| Temp | °C | 0.000 | 0 | 0 | 0 | 28.800 | 35.200 | Có | 348 | 0 | 0 | 0 |
| NH4+ | mg/L | 0.000 | 0 | 0 | 0 | 1.140 | 1.460 | Có | 224 | 144 | 0 | 56 |
| Color | Pt-Co | 0.031 | 2 | 2 | 0 | -2.700 | 53.300 | Có | 0 | 78 | 0 | 4 |
| Flow2 | m³/h | 0.000 | 0 | 0 | 0 | 25.242 | 143.343 | Có | 53 | 0 | 14 | 0 |
| Flow1 | m³/h | 0.000 | 0 | 0 | 0 | -52.787 | 223.873 | Có | 0 | 0 | 0 | 0 |

Ghi chú: COD **không** bị clip (mặc định `CLIP_COD = False`). Nếu áp ngưỡng IQR của train thì có 754 giá trị COD ở train và 0 giá trị ở test nằm ngoài ngưỡng.

## 2. Thống kê mô tả sau tiền xử lý

**Bảng 4. Thống kê mô tả 8 thông số sau tiền xử lý (toàn giai đoạn)** (`tables/descriptive_stats.csv`)

| Thông số | Đơn vị | N | Mean | Std | Min | Max | Q1 | Q3 | Skewness | Số giá trị bị clip |
|---|---|---|---|---|---|---|---|---|---|---|
| COD | mg/L | 6544 | 13.236 | 6.818 | 0.440 | 43.220 | 10.030 | 13.780 | 1.932 | 0 |
| TSS | mg/L | 6544 | 9.589 | 0.298 | 8.945 | 10.105 | 9.430 | 9.780 | -0.257 | 720 |
| pH | - | 6544 | 8.103 | 0.205 | 7.680 | 8.560 | 8.010 | 8.210 | -0.049 | 347 |
| Temp | °C | 6544 | 32.147 | 1.415 | 28.800 | 34.700 | 31.400 | 33.200 | -0.611 | 348 |
| NH4+ | mg/L | 6544 | 1.298 | 0.064 | 1.140 | 1.460 | 1.260 | 1.340 | 0.234 | 424 |
| Color | Pt-Co | 6544 | 26.204 | 9.493 | 0.000 | 53.300 | 19.100 | 32.400 | 0.454 | 82 |
| Flow2 | m³/h | 6544 | 83.523 | 23.287 | 25.242 | 114.180 | 73.333 | 101.410 | -1.000 | 67 |
| Flow1 | m³/h | 6544 | 91.471 | 49.983 | 0.000 | 158.570 | 57.065 | 132.960 | -0.695 | 0 |

**Bảng 5. Phân phối COD theo giai đoạn train/test** (`tables/cod_train_vs_test.csv`)

| Giai đoạn | count | mean | std | min | 25% | 50% | 75% | max | skew |
|---|---|---|---|---|---|---|---|---|---|
| Test | 1309 | 11.806 | 1.640 | 6.920 | 11.180 | 11.910 | 12.810 | 15.640 | -0.863 |
| Train | 5235 | 13.594 | 7.537 | 0.440 | 9.750 | 11.250 | 14.670 | 43.220 | 1.656 |

## 3. So sánh mô hình dự báo COD(t+1)

**Bảng 6. Kết quả trên tập test (n = 1307 cặp), trung bình ± độ lệch chuẩn qua 5 seed (42–46)** (`tables/model_metrics.csv`)

| Mô hình | R² | Adj R² | RMSE (mg/L) | MAE (mg/L) | MAPE (%) | Skill so với Persistence | Số seed |
|---|---|---|---|---|---|---|---|
| Persistence | 0.9779 ± 0.0000 | 0.9779 ± 0.0000 | 0.2430 ± 0.0000 | 0.1293 ± 0.0000 | 1.13 ± 0.00 | 0.0000 ± 0.0000 | 1 |
| LinearRegression | 0.9767 ± 0.0000 | 0.9765 ± 0.0000 | 0.2497 ± 0.0000 | 0.1442 ± 0.0000 | 1.27 ± 0.00 | -0.0276 ± 0.0000 | 1 |
| RandomForest | 0.8300 ± 0.0130 | 0.8289 ± 0.0131 | 0.6740 ± 0.0256 | 0.4768 ± 0.0166 | 3.97 ± 0.13 | -1.7737 ± 0.1054 | 5 |
| ExtraTrees | 0.8992 ± 0.0038 | 0.8986 ± 0.0038 | 0.5191 ± 0.0098 | 0.3964 ± 0.0144 | 3.41 ± 0.11 | -1.1361 ± 0.0403 | 5 |
| GradientBoosting | 0.6780 ± 0.0172 | 0.6760 ± 0.0173 | 0.9279 ± 0.0248 | 0.3591 ± 0.0064 | 2.95 ± 0.05 | -2.8184 ± 0.1019 | 5 |
| AdaBoost | 0.0406 ± 0.1192 | 0.0347 ± 0.1200 | 1.5995 ± 0.0988 | 1.4339 ± 0.1210 | 12.98 ± 0.99 | -5.5824 ± 0.4067 | 5 |
| XGBoost | 0.7860 ± 0.0000 | 0.7847 ± 0.0000 | 0.7565 ± 0.0000 | 0.6135 ± 0.0000 | 5.19 ± 0.00 | -2.1132 ± 0.0000 | 5 |
| CatBoost | 0.8265 ± 0.0567 | 0.8254 ± 0.0571 | 0.6746 ± 0.1069 | 0.5283 ± 0.1116 | 4.46 ± 0.93 | -1.7760 ± 0.4398 | 5 |
| Stacking | 0.1080 ± 0.2682 | 0.1025 ± 0.2698 | 1.5305 ± 0.2331 | 1.2538 ± 0.2089 | 10.68 ± 1.83 | -5.2984 ± 0.9591 | 5 |

Persistence và LinearRegression không có yếu tố ngẫu nhiên nên chỉ chạy 1 lần (std = 0). XGBoost với cấu hình mặc định (không subsample) cho kết quả giống hệt nhau ở mọi seed. Adj R² dùng p = 8 (p = 1 cho Persistence). Skill = 1 − RMSE_model / RMSE_Persistence.

**Bảng 7. Kiểm định Diebold–Mariano (hiệu chỉnh HLN, phương sai HAC Newey–West, lag = 7) và Wilcoxon signed-rank trên |sai số|, seed 42** (`tables/significance.csv`)

| So sánh (A vs B) | RMSE A | RMSE B | DM (MSE) | p DM (MSE) | DM (MAE) | p DM (MAE) | p Wilcoxon |e| | DM (MSE) 5 seed [min; max] | p DM (MSE) lớn nhất qua 5 seed |
|---|---|---|---|---|---|---|---|---|---|
| Stacking vs LinearRegression | 1.5746 | 0.2497 | 12.1297 | 3.67e-32 | 25.4913 | 1.14e-116 | 5.20e-208 | [8.77; 12.13] | 5.53e-18 |
| Stacking vs Persistence | 1.5746 | 0.2430 | 12.1522 | 2.87e-32 | 26.0333 | 1.07e-120 | 5.88e-209 | [8.79; 12.15] | 4.67e-18 |
| LinearRegression vs Persistence | 0.2497 | 0.2430 | 4.0838 | 4.70e-05 | 10.7821 | 5.02e-26 | 2.72e-37 | [4.08; 4.08] | 4.70e-05 |

DM < 0: mô hình A có sai số nhỏ hơn B; DM > 0: A kém hơn B. Wilcoxon không tính đến tự tương quan của sai số nên p-value của nó lạc quan hơn thực tế.

**Bảng 8. Hệ số của meta-learner (LinearRegression) trong Stacking theo seed** (`tables/meta_weights.csv`)

| Seed | Intercept | n_meta_samples | RandomForest | ExtraTrees | GradientBoosting | AdaBoost | XGBoost | CatBoost | Tổng hệ số |
|---|---|---|---|---|---|---|---|---|---|
| 42 | -6.0426 | 4355 | -0.2598 | 0.2670 | -0.7070 | 1.0625 | 0.6317 | 0.3319 | 1.3263 |
| 43 | -5.3789 | 4355 | -0.1836 | 0.5184 | -0.8353 | 1.1048 | 0.5347 | 0.1330 | 1.2719 |
| 44 | -5.0557 | 4355 | -0.1971 | 0.6077 | -0.7344 | 1.0474 | 0.5556 | -0.0261 | 1.2532 |
| 45 | -4.7904 | 4355 | -0.7085 | 1.1909 | -0.4572 | 1.0509 | 0.4970 | -0.3456 | 1.2275 |
| 46 | -6.3974 | 4355 | -0.1212 | 0.0493 | -0.8807 | 1.0799 | 0.5925 | 0.6225 | 1.3423 |

**Bảng 9. Phân tích bổ sung: các biến thể Stacking (seed 42)** (`tables/stacking_variants.csv`)

| Biến thể | R2 | RMSE | MAE | MAPE_% | Intercept meta | Tổng hệ số meta |
|---|---|---|---|---|---|---|
| TSS(5) + LinearRegression (mô hình chính) | 0.0730 | 1.5746 | 1.3416 | 11.4708 | -6.0426 | 1.3263 |
| KFold(5) + LinearRegression (phiên bản trước) | 0.5240 | 1.1284 | 0.6367 | 5.3334 | -1.9105 | 1.1653 |
| TSS(5) + LinearRegression(positive=True) | 0.3252 | 1.3435 | 1.2740 | 11.0944 | -5.2460 | 1.2690 |

**Bảng 10. Độ nhạy: clip IQR cả COD như phiên bản trước (seed 42, đánh giá trên COD thô)** (`tables/sensitivity_clip_cod.csv`)

| Model | R2 | RMSE | MAE | MAPE_% | RMSE (không clip COD, seed 42) |
|---|---|---|---|---|---|
| Persistence | 0.9779 | 0.2430 | 0.1293 | 1.1326 | 0.2430 |
| LinearRegression | 0.6278 | 0.9977 | 0.8010 | 7.1654 | 0.2497 |
| RandomForest | 0.8352 | 0.6639 | 0.4849 | 4.0519 | 0.6648 |
| ExtraTrees | 0.9023 | 0.5111 | 0.3768 | 3.2273 | 0.5292 |
| GradientBoosting | 0.7646 | 0.7935 | 0.3738 | 3.1151 | 0.9508 |
| AdaBoost | 0.0297 | 1.6110 | 1.3258 | 11.6259 | 1.4893 |
| XGBoost | 0.7081 | 0.8836 | 0.6909 | 5.6651 | 0.7565 |
| CatBoost | 0.7500 | 0.8178 | 0.6548 | 5.7049 | 0.5598 |
| Stacking | -1.0936 | 2.3664 | 1.8658 | 16.7562 | 1.5746 |

## 4. Thời gian huấn luyện và suy luận

**Bảng 11. Thời gian đo bằng `time.perf_counter`** (`tables/timing.csv`). Huấn luyện: median qua các seed; Stacking gồm cả 5 fold TimeSeriesSplit và lần huấn luyện lại trên toàn train. Suy luận: median của 7 lần dự báo toàn bộ tập test, chia cho số mẫu.

| Mô hình | Huấn luyện – median (s) | min (s) | max (s) | Suy luận (ms/mẫu) | Suy luận toàn test (s) | Số lần huấn luyện | n test |
|---|---|---|---|---|---|---|---|
| Persistence | 0.000002 | 0.000002 | 0.000002 | 0.000001 | 0.000002 | 1 | 1307 |
| LinearRegression | 0.006257 | 0.006257 | 0.006257 | 0.000691 | 0.000903 | 1 | 1307 |
| RandomForest | 3.754951 | 1.990825 | 4.966175 | 0.016005 | 0.020918 | 5 | 1307 |
| ExtraTrees | 1.226479 | 1.172325 | 2.832052 | 0.016330 | 0.021343 | 5 | 1307 |
| GradientBoosting | 1.045913 | 0.699724 | 1.523106 | 0.003275 | 0.004281 | 5 | 1307 |
| AdaBoost | 0.886526 | 0.439367 | 1.466205 | 0.038253 | 0.049996 | 5 | 1307 |
| XGBoost | 0.469623 | 0.201888 | 0.599172 | 0.002613 | 0.003415 | 5 | 1307 |
| CatBoost | 0.464337 | 0.259351 | 0.697376 | 0.002538 | 0.003318 | 5 | 1307 |
| Stacking | 33.326682 | 18.456775 | 43.151372 | 0.115442 | 0.150882 | 5 | 1307 |

## 5. Phân tích phần dư và trễ pha

**Bảng 12. Thống kê phần dư trên tập test (seed 42)** (`tables/residual_stats.csv`). Phần dư = thực tế − dự báo.

| Chỉ số | Stacking | XGBoost | CatBoost | ExtraTrees | LinearRegression | Persistence |
|---|---|---|---|---|---|---|
| n | 1307 | 1307 | 1307 | 1307 | 1307 | 1307 |
| Mean | 1.2751 | 0.3879 | 0.3087 | 0.1279 | -0.0183 | 0.0044 |
| Std | 0.9242 | 0.6497 | 0.4671 | 0.5137 | 0.2491 | 0.2430 |
| Skewness | 1.1995 | -1.1977 | -0.8568 | -0.4495 | -0.4703 | -0.6488 |
| Kurtosis (excess) | 8.3318 | 4.1718 | 6.3119 | 2.7509 | 21.4640 | 23.4895 |
| Shapiro_W | 0.8761 | 0.9373 | 0.9465 | 0.9738 | 0.7824 | 0.7466 |
| Shapiro_p | 5.18e-31 | 4.79e-23 | 2.29e-21 | 1.17e-14 | 1.52e-38 | 9.92e-41 |
| JarqueBera_stat | 4093.9107 | 1260.2585 | 2329.5413 | 456.1236 | 25137.3698 | 30139.4582 |
| JarqueBera_p | 0.00e+00 | 2.18e-274 | 0.00e+00 | 9.00e-100 | 0.00e+00 | 0.00e+00 |
| BP_LM (theo ŷ) | 94.8581 | 28.6565 | 31.0340 | 0.0009 | 0.6255 | 0.6196 |
| BP_p (theo ŷ) | 2.05e-22 | 8.64e-08 | 2.54e-08 | 0.9755 | 0.4290 | 0.4312 |
| BP_LM (theo 8 đặc trưng) | 108.0370 | 60.3341 | 93.4225 | 74.8667 | 34.4434 | 32.7485 |
| BP_p (theo 8 đặc trưng) | 9.64e-20 | 4.01e-10 | 9.37e-17 | 5.25e-13 | 3.38e-05 | 6.84e-05 |
| LjungBox_Q lag1 | 349.4717 | 309.1976 | 305.8995 | 489.0573 | 15.4192 | 36.2884 |
| LjungBox_p lag1 | 5.52e-78 | 3.27e-69 | 1.71e-68 | 2.29e-108 | 8.61e-05 | 1.70e-09 |
| ACF lag1 | 0.5165 | 0.4858 | 0.4832 | 0.6110 | -0.1085 | -0.1664 |

**Bảng 13. Tương quan chéo corr(ŷ_t, y_{t+k}) trên tập test** (`tables/lag_crosscorr.csv`). Nếu mô hình chỉ lặp lại giá trị hiện tại (trễ 1 bước) thì tương quan cực đại rơi vào k = −1.

| Mô hình | Chuỗi | k=-3 | k=-2 | k=-1 | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|---|---|---|---|
| LinearRegression | mức | 0.9817 | 0.9889 | 0.9997 | 0.9884 | 0.9807 | 0.9710 | 0.9608 |
| LinearRegression | sai phân | 0.0920 | -0.1636 | 0.9981 | -0.1650 | 0.0842 | 0.0275 | 0.0492 |
| Persistence | mức | 0.9815 | 0.9889 | 1.0000 | 0.9890 | 0.9816 | 0.9722 | 0.9620 |
| Persistence | sai phân | 0.0896 | -0.1666 | 1.0000 | -0.1664 | 0.0892 | 0.0367 | 0.0565 |
| Stacking | mức | 0.8512 | 0.8583 | 0.8662 | 0.8563 | 0.8469 | 0.8351 | 0.8234 |
| Stacking | sai phân | 0.0669 | -0.0130 | 0.2451 | -0.0202 | 0.0334 | -0.0031 | 0.0204 |

## 6. Tầm dự báo dài hơn

Đặc trưng: 8 thông số tại t + COD(t−1, t−2, t−3, t−6) + trung bình/độ lệch chuẩn trượt của COD (30 phút, 1 giờ) + COD(t) − COD(t−1); tất cả chỉ dùng quá khứ và tính trong từng đoạn liên tục.

**Bảng 14. Kết quả theo tầm dự báo (5 seed)** (`tables/horizon_metrics.csv`)

| Tầm dự báo | Mô hình | R² | RMSE (mg/L) | MAE (mg/L) | MAPE (%) | Skill so với Persistence | n test |
|---|---|---|---|---|---|---|---|
| 5 phút | Persistence | 0.9780 ± 0.0000 | 0.2430 ± 0.0000 | 0.1289 ± 0.0000 | 1.13 ± 0.00 | 0.0000 | 1296 |
| 5 phút | XGBoost | 0.8125 ± 0.0000 | 0.7102 ± 0.0000 | 0.4233 ± 0.0000 | 3.47 ± 0.00 | -1.9221 | 1296 |
| 5 phút | Stacking | 0.5444 ± 0.0492 | 1.1058 ± 0.0589 | 0.8632 ± 0.0753 | 7.47 ± 0.68 | -3.5500 | 1296 |
| 30 phút | Persistence | 0.8754 ± 0.0000 | 0.5720 ± 0.0000 | 0.2727 ± 0.0000 | 2.36 ± 0.00 | 0.0000 | 1286 |
| 30 phút | XGBoost | 0.7156 ± 0.0000 | 0.8642 ± 0.0000 | 0.6332 ± 0.0000 | 5.51 ± 0.00 | -0.5110 | 1286 |
| 30 phút | Stacking | -0.1825 ± 0.1815 | 1.7581 ± 0.1372 | 1.5973 ± 0.1582 | 13.76 ± 1.37 | -2.0737 | 1286 |
| 60 phút | Persistence | 0.7446 ± 0.0000 | 0.8085 ± 0.0000 | 0.4238 ± 0.0000 | 3.66 ± 0.00 | 0.0000 | 1274 |
| 60 phút | XGBoost | 0.5158 ± 0.0000 | 1.1131 ± 0.0000 | 0.7738 ± 0.0000 | 6.85 ± 0.00 | -0.3768 | 1274 |
| 60 phút | Stacking | -0.4724 ± 0.1250 | 1.9396 ± 0.0826 | 1.7265 ± 0.0941 | 14.42 ± 0.87 | -1.3992 | 1274 |
| 120 phút | Persistence | 0.5011 ± 0.0000 | 1.0936 ± 0.0000 | 0.6537 ± 0.0000 | 5.71 ± 0.00 | 0.0000 | 1250 |
| 120 phút | XGBoost | 0.3258 ± 0.0000 | 1.2713 ± 0.0000 | 0.8927 ± 0.0000 | 7.81 ± 0.00 | -0.1625 | 1250 |
| 120 phút | Stacking | -1.2660 ± 0.5501 | 2.3160 ± 0.2921 | 2.0540 ± 0.3193 | 17.30 ± 2.64 | -1.1178 | 1250 |

**Bảng 15. Diebold–Mariano theo tầm dự báo (seed 42)** (`tables/horizon_significance.csv`)

| Horizon_steps | So sánh | DM_stat (MSE) | DM_p (MSE) | HAC_lag |
|---|---|---|---|---|
| 1 | Stacking vs Persistence | 12.0185 | 1.27e-31 | 7 |
| 1 | XGBoost vs Persistence | 3.8016 | 1.50e-04 | 7 |
| 1 | Stacking vs XGBoost | 6.9507 | 5.75e-12 | 7 |
| 6 | Stacking vs Persistence | 17.7028 | 6.37e-63 | 7 |
| 6 | XGBoost vs Persistence | 9.5706 | 5.23e-21 | 7 |
| 6 | Stacking vs XGBoost | 13.9781 | 1.95e-41 | 7 |
| 12 | Stacking vs Persistence | 12.1408 | 3.57e-32 | 11 |
| 12 | XGBoost vs Persistence | 4.5755 | 5.22e-06 | 11 |
| 12 | Stacking vs XGBoost | 9.0433 | 5.53e-19 | 11 |
| 24 | Stacking vs Persistence | 6.5474 | 8.53e-11 | 23 |
| 24 | XGBoost vs Persistence | 1.7664 | 0.0776 | 23 |
| 24 | Stacking vs XGBoost | 6.9153 | 7.43e-12 | 23 |

## 7. Giải thích mô hình bằng SHAP

SHAP tính trên 500/1307 mẫu test (lấy mẫu ngẫu nhiên, seed 42); nền = 200 mẫu ngẫu nhiên từ train (seed 42).
Thời gian tính SHAP Stacking (exact): 599.1 s; sai số cộng tính lớn nhất: 7.28e-14 mg/L

**Bảng 16. Trung bình |SHAP| (mg/L) theo mô hình** (`tables/shap_importance.csv`)

| Đặc trưng | CatBoost | Stacking | XGBoost |
|---|---|---|---|
| COD | 2.7425 | 3.4573 | 2.1764 |
| pH | 0.3814 | 0.5310 | 0.0989 |
| Temp | 0.2402 | 0.4213 | 0.2532 |
| Flow2 | 0.3961 | 0.2497 | 0.1424 |
| Flow1 | 0.2648 | 0.2208 | 0.1032 |
| TSS | 0.1589 | 0.1277 | 0.0670 |
| Color | 0.3333 | 0.1179 | 0.1023 |
| NH4+ | 0.1273 | 0.0728 | 0.1167 |

**Bảng 17. Chiều tác động của từng đặc trưng trong Stacking.** Spearman(giá trị, SHAP) > 0 nghĩa là giá trị đặc trưng càng cao thì đóng góp vào dự báo càng lớn.

| Đặc trưng | Hạng | Mean|SHAP| (mg/L) | Mean SHAP (mg/L) | Spearman(giá trị, SHAP) |
|---|---|---|---|---|
| COD | 1 | 3.4573 | -3.4573 | 0.9553 |
| pH | 2 | 0.5310 | 0.3196 | -0.8149 |
| Temp | 3 | 0.4213 | -0.3783 | -0.8467 |
| Flow2 | 4 | 0.2497 | -0.0385 | 0.1508 |
| Flow1 | 5 | 0.2208 | 0.1177 | 0.5370 |
| TSS | 6 | 0.1277 | 0.0311 | -0.1377 |
| Color | 7 | 0.1179 | -0.0129 | 0.3378 |
| NH4+ | 8 | 0.0728 | 0.0605 | -0.0126 |

## 8. Danh sách hình (outputs/v2/figures)

- **Hình 1** (`figures/fig01_correlation_heatmap.png`): Ma trận tương quan Pearson giữa 8 thông số tại thời điểm t và COD(t+1), tính trên 6538 cặp (t, t+1) dùng cho train và test.
- **Hình 2** (`figures/fig02_timeseries_overview.png`): Chuỗi COD quan trắc 01–23/01/2026 (chu kỳ 5 phút); nền xanh là giai đoạn train (80%), nền cam là giai đoạn test (20%); đường đứt là ngưỡng IQR trên của COD tính trên train.
- **Hình 3** (`figures/fig03_model_comparison.png`): RMSE trên tập test của các mô hình (trung bình ± độ lệch chuẩn qua 5 seed).
- **Hình 4** (`figures/fig04_actual_vs_predicted.png`): COD thực tế, dự báo của Stacking và của Persistence trong 150 chu kỳ đầu của tập test.
- **Hình 5** (`figures/fig05_scatter_actual_pred.png`): COD thực tế và dự báo trên tập test của Stacking, LinearRegression (mô hình đơn tốt nhất) và Persistence; đường đứt là y = x.
- **Hình 6** (`figures/fig06_residuals.png`): Phần dư của Stacking trên tập test: (a) histogram, (b) phần dư theo giá trị dự báo kèm đường LOWESS, (c) biểu đồ Q-Q so với phân phối chuẩn.
- **Hình 7** (`figures/fig07_lag_crosscorr.png`): Tương quan chéo corr(ŷ_t, y_t+k) với k = −3…+3 giữa dự báo và thực tế của Stacking, LinearRegression và Persistence, cho chuỗi mức (trái) và chuỗi sai phân bậc 1 (phải).
- **Hình 8** (`figures/fig08_shap_bar.png`): Trung bình |SHAP| (mg/L) của từng đặc trưng trong mô hình Stacking.
- **Hình 9** (`figures/fig09_shap_beeswarm.png`): Phân bố giá trị SHAP của mô hình Stacking; màu thể hiện giá trị đặc trưng.
- **Hình 10** (`figures/fig10_shap_compare.png`): So sánh trung bình |SHAP| giữa Stacking (Exact), XGBoost và CatBoost (TreeSHAP) trên cùng tập mẫu test.
- **Hình 11** (`figures/fig11_shap_dependence.png`): Biểu đồ phụ thuộc SHAP của COD(t), nhiệt độ và NH4+ trong mô hình Stacking (đường đỏ là LOWESS).
- **Hình 12** (`figures/fig12_horizon_rmse.png`): RMSE trên tập test theo tầm dự báo (5 phút, 30 phút, 1 giờ, 2 giờ) của Persistence, XGBoost và Stacking (± std qua 5 seed).
- **Hình 13** (`figures/fig13_workflow.png`): Sơ đồ quy trình xử lý dữ liệu, huấn luyện, đánh giá và giải thích mô hình.

## 9. Nhận xét trung thực

**(1) Stacking có tốt hơn Persistence và mô hình đơn tốt nhất không?**

- Persistence (ŷ(t+1) = COD(t)) đạt RMSE = 0.2430 mg/L, R² = 0.9779. Mô hình đơn tốt nhất theo RMSE trung bình là **LinearRegression** (RMSE = 0.2497 ± 0.0000). Stacking đạt RMSE = 1.5305 ± 0.2331, R² = 0.1080 ± 0.2682.
- Stacking **kém hơn** Persistence (Skill = -5.2984) và kém hơn LinearRegression. Ở seed 42, DM (MSE) = 12.15 với p = 2.87e-32 (so với Persistence) và DM = 12.13 với p = 3.67e-32 (so với LinearRegression). p-value lớn nhất qua 5 seed lần lượt là 4.67e-18 và 5.53e-18, nên **chênh lệch có ý nghĩa thống kê theo hướng Stacking kém hơn**.
- Không mô hình học máy nào có RMSE thấp hơn Persistence. 6/6 mô hình cây có RMSE cao hơn Persistence; LinearRegression có RMSE = 0.2497 (Skill = -0.0276). Chênh lệch giữa LinearRegression và Persistence nhỏ về độ lớn (0.0067 mg/L) nhưng có ý nghĩa thống kê theo DM (DM = 4.08, p = 4.70e-05; DM > 0 nghĩa là LinearRegression kém hơn).
- Vì vậy R² cao (≈ 0.978) ở bài toán t+1 **chủ yếu đến từ quán tính của chuỗi COD** (COD(t) ≈ COD(t+1)), không phải năng lực riêng của mô hình. Không thể khẳng định ‘Stacking là mô hình tốt nhất’ như phiên bản trước.

**(2) Vì sao Stacking kém?** (các phân tích dưới đây là bằng chứng có trong số liệu, không phải kết luận nhân quả chắc chắn)

- Meta-learner học trên dự báo OOF của TimeSeriesSplit; các fold kiểm định này nằm trong giai đoạn train, giai đoạn có các đợt COD tăng vọt (train: std = 7.54 mg/L, max = 43.22), trong khi giai đoạn test yên hơn nhiều (std = 1.64, max = 15.64). Hệ số meta có tổng từ 1.228 đến 1.342 và intercept từ -6.397 đến -4.790 (Bảng 8), có cả hệ số âm. Khi áp lên test, bộ hệ số này làm dự báo lệch có hệ thống: trung bình phần dư của Stacking = 1.2751 mg/L, so với 0.0044 của Persistence.
- Bảng 9: nếu dùng KFold như phiên bản trước, RMSE = 1.1284; nếu ràng buộc hệ số meta ≥ 0, RMSE = 1.3435; mô hình chính (TimeSeriesSplit) có RMSE = 1.5746. Kết quả của Stacking vì vậy rất nhạy với cách tạo OOF. Kể cả biến thể tốt nhất cũng không vượt Persistence.
- Bản thân các mô hình cơ sở đã kém Persistence (ví dụ ExtraTrees: RMSE = 0.5191, gấp 2.14 lần) và có phần dư trung bình khác 0 (XGBoost 0.3879, CatBoost 0.3087 mg/L), nên không thể kỳ vọng tổ hợp tuyến tính của chúng vượt Persistence. Nguyên nhân cụ thể khiến các mô hình cây kém (ví dụ xấp xỉ quan hệ gần như đồng nhất COD(t) → COD(t+1) bằng hàm bậc thang, hoặc phụ thuộc vào các cảm biến khác) **chưa được kiểm chứng riêng** trong đồ án này.

**(3) Phần dư có chuẩn và đồng phương sai không?**

- Phần dư Stacking: mean = 1.2751, std = 0.9242, skewness = 1.200, kurtosis dư = 8.332. Shapiro–Wilk p = 5.18e-31 và Jarque–Bera p = 0.00e+00, nên **bác bỏ giả thuyết phân phối chuẩn**.
- Breusch–Pagan theo ŷ: p = 2.05e-22; theo 8 đặc trưng: p = 9.64e-20. Có bằng chứng **phương sai sai số thay đổi** (không đồng phương sai) ở mức 5%.
- Ljung–Box lag 1: p = 5.52e-78, ACF(1) = 0.516. Phần dư **tự tương quan mạnh**, tức mô hình còn bỏ sót cấu trúc thời gian. Persistence và LinearRegression cũng có phần dư không chuẩn (kurtosis dư 23.5 và 21.5) do các bước nhảy đột ngột của COD.

**(4) Mô hình có bị trễ pha không?**

- Persistence: chuỗi mức đạt tương quan cực đại ở k = -1 (1.0000; tại k = 0 là 0.9890); chuỗi sai phân đạt cực đại ở k = -1 (1.0000; tại k = 0 là -0.1664).
- LinearRegression: chuỗi mức đạt tương quan cực đại ở k = -1 (0.9997; tại k = 0 là 0.9884); chuỗi sai phân đạt cực đại ở k = -1 (0.9981; tại k = 0 là -0.1650).
- Stacking: chuỗi mức đạt tương quan cực đại ở k = -1 (0.8662; tại k = 0 là 0.8563); chuỗi sai phân đạt cực đại ở k = -1 (0.2451; tại k = 0 là -0.0202).
- Cực đại ở k = −1 nghĩa là dự báo tại t giống nhất với giá trị thực tế của chu kỳ **trước**, tức **trễ 1 bước** (dự báo về cơ bản lặp lại COD hiện tại). Kiểm tra trên chuỗi sai phân rõ hơn vì loại bỏ thành phần mức chung.
- Các mô hình có cực đại sai phân ở k = −1: Persistence, LinearRegression, Stacking. LinearRegression có tương quan sai phân 0.9981 tại k = −1 nhưng -0.1650 tại k = 0, nên về bản chất nó đang mô phỏng Persistence và **chưa dự báo được thời điểm COD đổi hướng**. Stacking cũng trễ 1 bước, nhưng tương quan sai phân chỉ 0.2451, tức dự báo của nó còn thêm dao động riêng không khớp với thay đổi thực tế.

**(5) Biến quan trọng nhất theo SHAP và chiều tác động (mô hình Stacking)**

- Biến quan trọng nhất: **COD(t)** với mean|SHAP| = 3.4573 mg/L, gấp 6.5 lần biến thứ hai (pH: 0.5310). Spearman(giá trị, SHAP) của COD = 0.955, tức COD hiện tại càng cao thì dự báo càng cao.
- Các biến còn lại (mean|SHAP| mg/L; ρ Spearman): pH (0.531; ρ = -0.81), Temp (0.421; ρ = -0.85), Flow2 (0.250; ρ = +0.15), Flow1 (0.221; ρ = +0.54), TSS (0.128; ρ = -0.14), Color (0.118; ρ = +0.34), NH4+ (0.073; ρ = -0.01).
- Thứ hạng COD trong XGBoost / CatBoost: 1 / 1; cả ba mô hình đều đặt COD(t) đứng đầu. Thứ tự các biến phụ khác nhau giữa các mô hình (Bảng 16), nên **không nên diễn giải thứ tự của các biến phụ như quan hệ nhân quả**.
- Lưu ý: SHAP giải thích **mô hình**, không phải quá trình vật lý. Base value của Stacking (13.903 mg/L) được tính trên nền train, nên giá trị SHAP âm của COD ở vùng COD thấp phản ánh việc test nằm dưới mức trung bình của train.

**(6) Tầm dự báo dài hơn (cảnh báo sớm)**

- 5 phút: RMSE Persistence = 0.2430, XGBoost = 0.7102 (Skill -1.922; DM p = 1.50e-04), Stacking = 1.1058 (Skill -3.550; DM p = 1.27e-31).
- 30 phút: RMSE Persistence = 0.5720, XGBoost = 0.8642 (Skill -0.511; DM p = 5.23e-21), Stacking = 1.7581 (Skill -2.074; DM p = 6.37e-63).
- 60 phút: RMSE Persistence = 0.8085, XGBoost = 1.1131 (Skill -0.377; DM p = 5.22e-06), Stacking = 1.9396 (Skill -1.399; DM p = 3.57e-32).
- 120 phút: RMSE Persistence = 1.0936, XGBoost = 1.2713 (Skill -0.163; DM p = 0.0776), Stacking = 2.3160 (Skill -1.118; DM p = 8.53e-11).
- So sánh **không** có ý nghĩa thống kê ở mức 5% (DM, seed 42): XGBoost vs Persistence ở 120 phút (p = 0.0776). Ở các trường hợp này chỉ có thể nói hai mô hình không khác biệt rõ.
- Ở **mọi tầm dự báo đã thử**, cả XGBoost lẫn Stacking đều không vượt Persistence trên tập test này. Vì vậy **chưa có cơ sở để nói hệ thống ‘cảnh báo sớm 1–2 giờ’**.

**(7) Hạn chế**

- Chỉ có 1 trạm, 22.9062 ngày dữ liệu; tập test chỉ 1307 cặp (khoảng 4,5 ngày) và rơi vào giai đoạn COD ổn định (max 15.64 mg/L). Mô hình chưa được kiểm chứng trên các đợt COD tăng cao.
- Chỉ dùng một lần chia train/test cố định. Nên bổ sung đánh giá rolling-origin (walk-forward) trên nhiều giai đoạn để có ước lượng ổn định hơn.
- Siêu tham số giữ mặc định, chưa tinh chỉnh. Kiểm định DM giả định chuỗi chênh lệch sai số dừng.
- SHAP của Stacking tính trên mẫu con test (xem Mục 7). TreeSHAP của XGBoost/CatBoost dùng chế độ path-dependent, khác chế độ interventional của Stacking, nên chỉ nên so sánh thứ hạng, không so sánh trực tiếp độ lớn.
- Thời gian đo trên một máy (xem `environment.txt`). XGBoost/CatBoost dùng đa luồng còn sklearn dùng 1 luồng, nên so sánh tốc độ giữa các họ mô hình chỉ mang tính tham khảo. Stacking huấn luyện mất 33.3 s (median) và suy luận 0.1154 ms/mẫu.

## 10. Những thay đổi so với phiên bản cũ

1. **Đọc dữ liệu:** đọc thêm cột trạng thái; giá trị có status ≠ `00` (2 giá trị Color mang mã `01`) được coi là thiếu.
2. **Trục thời gian:** kiểm tra trùng lặp/gap và chia chuỗi thành các đoạn liên tục. Chỉ tạo cặp (t, t+h) khi hai thời điểm cách nhau đúng h·5 phút (bỏ 4 cặp từng nối qua gap). Các cặp train có mục tiêu rơi vào giai đoạn test bị loại (purge).
3. **Chống rò rỉ:** chia train/test (80/20 theo thời gian) **trước**; median và ngưỡng IQR chỉ fit trên train rồi áp dụng cho test. MinMaxScaler nằm trong Pipeline, chỉ fit trên train.
4. **Điền thiếu:** forward-fill trong cùng đoạn liên tục (chỉ dùng quá khứ, không nối qua gap); nếu còn thiếu (ở đầu đoạn) thì dùng median của train. Không dùng nội suy hai phía, vì nó dùng giá trị tương lai.
5. **IQR:** vẫn ở dạng clip (không xoá dòng) cho 7 cảm biến; **không clip COD** vì COD(t+1) là biến mục tiêu. Bản cũ clip cả COD về [4,40; 19,41] mg/L, làm mất các đỉnh ô nhiễm. Mức ảnh hưởng xem Bảng 10.
6. **Baseline:** thêm Persistence (ŷ(t+1) = COD(t)) và LinearRegression trên 8 đặc trưng.
7. **Stacking:** thay `cv=5` (KFold) bằng TimeSeriesSplit(5). Vì `StackingRegressor` của sklearn không nhận TimeSeriesSplit, đã tự cài `TimeSeriesStackingRegressor` (src/ensemble_models.py): meta-learner học trên dự báo OOF của các fold (chỉ dùng quá khứ), sau đó huấn luyện lại các mô hình cơ sở trên toàn bộ train.
8. **Đánh giá:** thêm MAPE, Skill so với Persistence; chạy 5 seed (42–46) và báo cáo mean ± std; kiểm định Diebold–Mariano (HLN + HAC) và Wilcoxon; đo thời gian bằng `time.perf_counter`.
9. **SHAP:** giải thích **chính mô hình Stacking** ở thang mg/L gốc bằng `shap.Explainer(algorithm='exact')` (nền 200 mẫu train), thay vì ExtraTrees trên dữ liệu đã scale. Thêm TreeSHAP của XGBoost/CatBoost để đối chiếu.
10. **Tầm dự báo:** thêm thí nghiệm h = 5, 30, 60, 120 phút với đặc trưng trễ và trượt của COD.
11. **Ứng dụng:** thanh trượt giới hạn trong khoảng train, có cảnh báo khi nhập ngoài khoảng; ngưỡng theo QCVN 40:2025/BTNMT nằm trong dict cấu hình (cần điền); SHAP waterfall cho đúng mô hình đang dự báo; mặc định chọn mô hình học máy có RMSE test thấp nhất.

**Bảng 18. Đối chiếu số liệu cũ (`tables/previous_model_metrics.csv`) và mới.** Số cũ được tính khi target đã bị clip, cặp t→t+1 nối qua gap, Stacking dùng KFold và chỉ chạy 1 seed, nên không so sánh trực tiếp được; bảng này chỉ để thấy mức độ thay đổi.

| Mô hình | R² cũ | RMSE cũ | R² mới (mean) | RMSE mới (mean) |
|---|---|---|---|---|
| RandomForest | 0.8433 | 0.6490 | 0.8300 | 0.6740 |
| ExtraTrees | 0.8995 | 0.5196 | 0.8992 | 0.5191 |
| GradientBoosting | 0.8941 | 0.5334 | 0.6780 | 0.9279 |
| AdaBoost | 0.7009 | 0.8966 | 0.0406 | 1.5995 |
| XGBoost | 0.8277 | 0.6803 | 0.7860 | 0.7565 |
| CatBoost | 0.8746 | 0.5805 | 0.8265 | 0.6746 |
| Stacking | 0.9009 | 0.5159 | 0.1080 | 1.5305 |

Persistence (không có trong phiên bản trước): R² = 0.9779, RMSE = 0.2430.
