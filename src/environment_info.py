"""Ghi phiên bản thư viện và cấu hình máy vào outputs/v2/environment.txt."""
import os
import sys
import ctypes
import platform
import importlib
import datetime

try:
    from . import config as C
except ImportError:
    import config as C


def cpu_name():
    if platform.system() == 'Windows':
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or 'không xác định'


def ram_gb():
    if platform.system() == 'Windows':
        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1024 ** 3
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return float('nan')


def write_environment():
    libs = ['numpy', 'pandas', 'sklearn', 'xgboost', 'catboost', 'shap', 'scipy',
            'statsmodels', 'matplotlib', 'seaborn', 'streamlit', 'joblib']
    lines = [f"Thời điểm ghi: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
             f"OS: {platform.platform()}",
             f"CPU: {cpu_name()}",
             f"Số luồng logic (os.cpu_count): {os.cpu_count()}",
             f"RAM: {ram_gb():.1f} GB",
             f"Python: {sys.version.split()[0]}",
             f"Seed chính: {C.SEED}; các seed: {C.SEEDS}",
             "Ghi chú luồng: sklearn mặc định 1 luồng (n_jobs=None); XGBoost và CatBoost mặc định dùng mọi luồng.",
             "", "Phiên bản thư viện:"]
    for name in libs:
        try:
            lines.append(f"  {name}=={importlib.import_module(name).__version__}")
        except ImportError:
            lines.append(f"  {name}: chưa cài")
    path = os.path.join(C.OUT_DIR, "environment.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    write_environment()
