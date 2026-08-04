import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DIR = DATA_DIR / "raw"
SMP_DIR = RAW_DIR / "market"
HYDRO_DIR = RAW_DIR / "hydro"
MISC_DIR = RAW_DIR / "exogenous"

MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PROCESSED_DIR = OUTPUT_DIR / "processed"

for d in (MODEL_DIR, FIGURE_DIR, METRICS_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

SMP_FILE = SMP_DIR / "smp_prices_nsmo.csv"
LOAD_FILE = SMP_DIR / "load_data_nsmo.csv"
DISPATCH_FILE = SMP_DIR / "dispatch_capacity_nsmo.csv"
WEATHER_FILE = MISC_DIR / "weather_3_regions_30min_open_meteo.csv"
FUEL_FILE = MISC_DIR / "fuel_macro_yfinance_clean.csv"
CALENDAR_FILE = MISC_DIR / "calendar_vietnam.csv"
DESIGN_MATRIX_FILE = PROCESSED_DIR / "design_matrix.parquet"

TIMELINE_START = "2021-01-01 00:00:00"
TIMELINE_FREQ = "30min"
CYCLES_PER_DAY = 48

PRICE_CAP_VND = 1778.6
NEAR_ZERO_THRESHOLD = 2.0

SMP_LAGS = [48, 49, 50, 96, 336]
LOAD_LAGS = [48, 49, 50, 96, 336]
REGIONAL_LAGS = [48, 336]
ROLLING_WINDOWS = {"24h": 48, "72h": 144}
LAG_SAFETY_SHIFT = 48

CALENDAR_COLS = [
    "is_weekend", "is_workday", "is_holiday",
    "is_tet", "is_pre_holiday", "is_post_holiday", "season",
]

WEATHER_REGIONS = ["hanoi", "danang", "hcmc"]
WEATHER_VARS = [
    "temperature", "humidity", "cloud_cover", "wind_speed",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
]

DISPATCH_RESOURCE_PATTERNS = {
    "total": "quốc",
    "hydro": "Thủy",
    "solar": "trời trang",
    "wind": "gió",
}

FUEL_COLS = [
    "coal_proxy_price", "brent_price", "gas_proxy_price",
    "usd_vnd", "dxy_index",
]

CV_SPLITS = [
    {"name": "fold_1", "train_end": "2024-06-30", "val_start": "2024-07-01", "val_end": "2025-03-31"},
    {"name": "fold_2", "train_end": "2025-06-30", "val_start": "2025-07-01", "val_end": "2026-03-31"},
    {"name": "final_test", "train_end": "2026-03-31", "val_start": "2026-04-01", "val_end": "2026-06-19"},
]

LGB_PARAMS = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
LGB_NUM_BOOST_ROUND = 1000
LGB_EARLY_STOPPING = 50
