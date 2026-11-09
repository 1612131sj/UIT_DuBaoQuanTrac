"""Kiểu vẽ chung cho hình báo cáo: PNG 300 dpi, rộng ~16 cm, font có dấu tiếng Việt."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CM = 1 / 2.54
WIDTH = 16 * CM           # chiều rộng chèn khổ A4

# Bảng màu phân loại (thứ tự cố định, không xoay vòng)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'

# Màu cố định theo mô hình (màu đi theo thực thể, không theo thứ hạng)
MODEL_COLORS = {'Thực tế': INK, 'Stacking': BLUE, 'Persistence': ORANGE, 'XGBoost': AQUA,
                'CatBoost': VIOLET, 'LinearRegression': MUTED}


def setup():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.4, 'savefig.dpi': 300, 'figure.dpi': 100,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05, 'axes.unicode_minus': False,
    })


def save(fig, path):
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)
    print("  ->", path)
