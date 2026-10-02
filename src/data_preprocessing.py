import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def parse_txt_file(filepath: str) -> dict:
    """
    Đọc 1 file .txt thô và chuyển thành 1 dictionary (tương ứng với 1 dòng dữ liệu).
    File có dạng tab-separated, mỗi dòng là 1 thông số:
    COD	14.67	mg/L	20260101000500	00
    """
    filename = os.path.basename(filepath)
    # Tên file có dạng: <Mã trạm>_<ThoiDiemQuanTrac>.txt
    # VD: LA_KCN_ANHHON_20260101000500.txt
    name_without_ext = filename.replace('.txt', '')
    
    # Tách mã trạm và thời gian. Thời gian luôn là chuỗi số ở cuối cùng (sau dấu _ cuối)
    parts = name_without_ext.split('_')
    timestamp_str = parts[-1] 
    station_code = '_'.join(parts[:-1])
    
    record = {
        'Station': station_code,
        'Timestamp': timestamp_str
    }
    
    try:
        # Đọc file tab-separated, không có header
        # Cột: [0] Parameter, [1] Value, [2] Unit, [3] Timestamp, [4] Status
        df = pd.read_csv(filepath, sep='\t', header=None, engine='python', on_bad_lines='skip')
        
        # Đưa từng thông số vào record
        for _, row in df.iterrows():
            if len(row) >= 2:
                param_name = str(row[0]).strip()
                try:
                    param_value = float(row[1])
                except ValueError:
                    param_value = np.nan
                record[param_name] = param_value
                
    except Exception as e:
        print(f"Lỗi đọc file {filename}: {e}")
        
    return record

def merge_raw_data(raw_data_dir: str) -> pd.DataFrame:
    """
    Quét đệ quy toàn bộ các file .txt trong các thư mục con và gộp lại.
    """
    # Tìm TẤT CẢ các file .txt trong mọi thư mục con (recursive=True)
    search_pattern = os.path.join(raw_data_dir, "**", "*.txt")
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy file .txt nào trong thư mục {raw_data_dir} (hoặc các thư mục con)")
        
    print(f"Đã tìm thấy {len(all_files)} file .txt. Đang xử lý...")
    
    records = []
    for file in all_files:
        record = parse_txt_file(file)
        records.append(record)
            
    # Tạo DataFrame từ list các dictionary
    merged_df = pd.DataFrame(records)
    
    # Chuyển đổi cột Timestamp sang định dạng datetime cho chuẩn
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], format='%Y%m%d%H%M%S', errors='coerce')
    
    # Sắp xếp theo trạm và thời gian
    merged_df = merged_df.sort_values(by=['Station', 'Timestamp']).reset_index(drop=True)
    
    return merged_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện:
    1. Điền giá trị thiếu (Median Imputation)
    2. Lọc nhiễu bằng IQR
    """
    print("Bắt đầu tiền xử lý dữ liệu...")
    df_clean = df.copy()
    
    # Chỉ lấy các cột chứa số (bỏ qua Station và Timestamp)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Điền giá trị thiếu (Median Imputation)
    print("- Đang điền giá trị thiếu (Median Imputation)...")
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        
    # 2. Xử lý nhiễu bằng IQR
    # (Ở đây mình chọn cách "Clip" - giới hạn giá trị nhiễu ở mức Min/Max của IQR 
    # thay vì xóa hẳn dòng, để không làm đứt gãy chuỗi thời gian)
    print("- Đang xử lý nhiễu (IQR Capping)...")
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Đưa các giá trị vượt ngưỡng về lại ngưỡng an toàn
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
    print("Tiền xử lý hoàn tất!")
    return df_clean

if __name__ == "__main__":
    # Lấy đường dẫn thư mục gốc của dự án (thư mục chứa thư mục src)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # 1. Gộp dữ liệu thô
    df_raw = merge_raw_data(RAW_DIR)
    print("\n[1] Kích thước dữ liệu gốc:", df_raw.shape)
    
    # 2. Làm sạch dữ liệu
    df_clean = preprocess_data(df_raw)
    print("\n[2] Dữ liệu sau khi làm sạch:")
    print(df_clean.describe()) # In ra thống kê để kiểm tra
    
    # 3. Lưu file đã gộp & làm sạch để train mô hình
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "water_quality_cleaned.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"\n[3] Đã lưu file sạch tại: {out_path}")
