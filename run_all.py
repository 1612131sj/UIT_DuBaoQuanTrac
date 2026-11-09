"""
Chạy toàn bộ pipeline v2:   python run_all.py
Kết quả: outputs/v2/ (tables/, figures/, environment.txt, RESULTS.md), mô hình: models/v2/.
Tuỳ chọn: python run_all.py --skip-horizon --skip-shap  (bỏ qua các bước chạy lâu)
"""
import os
import sys
import time
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import config as C

np.random.seed(C.SEED)


def step(name, fn):
    print(f"\n{'=' * 70}\n▶ {name}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    fn()
    print(f"✓ {name}: {time.perf_counter() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-horizon', action='store_true')
    ap.add_argument('--skip-shap', action='store_true')
    args = ap.parse_args()

    import environment_info, data_preprocessing, train_pipeline
    import horizon_experiment, residual_plots, generate_paper_plots, explainability, make_results

    step("Ghi thông tin môi trường", environment_info.write_environment)
    step("Kiểm tra dữ liệu & tiền xử lý", data_preprocessing.run_audit_and_stats)
    step("Huấn luyện, baseline, kiểm định, thời gian", train_pipeline.main)
    if not args.skip_horizon:
        step("Tầm dự báo dài hơn", horizon_experiment.main)
    step("Phân tích phần dư", residual_plots.main)
    step("Vẽ hình báo cáo", generate_paper_plots.main)
    if not args.skip_shap:
        step("Giải thích mô hình (SHAP)", explainability.main)
    step("Tổng hợp RESULTS.md", make_results.main)


if __name__ == "__main__":
    main()
