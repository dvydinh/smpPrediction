import pandas as pd
import numpy as np
import glob
import gc
from pathlib import Path

def get_data_paths():
    KAGGLE_INPUT = Path('/kaggle/input')
    if KAGGLE_INPUT.exists():
        found_market = list(KAGGLE_INPUT.rglob("market/smp_prices_nsmo.csv"))
        DATA_ROOT = found_market[0].parent.parent if found_market else (list(KAGGLE_INPUT.iterdir())[0] if list(KAGGLE_INPUT.iterdir()) else Path("."))
        OUTPUT_DIR = Path('/kaggle/working')
    else:
        DATA_ROOT = Path('data/raw')
        OUTPUT_DIR = Path('outputs')
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT, OUTPUT_DIR

def load_and_preprocess_data(DATA_ROOT):
    MARKET_DIR = DATA_ROOT / 'market'
    HYDRO_DIR  = DATA_ROOT / 'hydro'
    EXOG_DIR   = DATA_ROOT / 'exogenous'
    
    master_idx = pd.date_range('2021-01-01', '2026-06-19 23:30', freq='30min')
    df = pd.DataFrame(index=master_idx)
    df.index.name = 'datetime'

    smp = pd.read_csv(MARKET_DIR / 'smp_prices_nsmo.csv')
    smp['datetime'] = pd.to_datetime(smp['datetime'])
    smp = smp.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    for col in ['smp_system_price', 'smp_north_price', 'smp_central_price', 'smp_south_price']:
        df[col] = smp[col]
    df['cycle_id'] = df.index.hour * 2 + df.index.minute // 30

    load = pd.read_csv(MARKET_DIR / 'load_data_nsmo.csv')
    load['datetime'] = pd.to_datetime(load['datetime'])
    load = load.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    for col in ['load_total_mw', 'load_north_mw', 'load_central_mw', 'load_south_mw']:
        df[col] = load[col]
    del smp, load; gc.collect()

    hydro_files = sorted(glob.glob(str(HYDRO_DIR / 'hydro_hourly_*.csv')))
    hydro_chunks = []
    for f in hydro_files:
        h = pd.read_csv(f)
        h['datetime'] = pd.to_datetime(h['datetime'])
        agg = h.groupby('datetime').agg({
            'inflow_m3s': 'sum', 'total_discharge_m3s': 'sum',
            'plant_discharge_m3s': 'sum', 'spill_discharge_m3s': 'sum',
            'water_level_m': 'mean',
        })
        hydro_chunks.append(agg)
    hydro = pd.concat(hydro_chunks).groupby(level=0).mean()
    hydro.columns = ['hydro_' + c for c in hydro.columns]
    hydro = hydro.sort_index()
    idx_30 = pd.date_range(hydro.index.min(), hydro.index.max() + pd.Timedelta('30min'), freq='30min')
    hydro = hydro.reindex(idx_30).ffill()
    for col in hydro.columns:
        df[col] = hydro[col]
    del hydro_chunks, hydro; gc.collect()

    disp = pd.read_csv(MARKET_DIR / 'dispatch_capacity_nsmo.csv', encoding='utf-8')
    disp['date'] = pd.to_datetime(disp['date'])
    patterns = {'total': 'quốc', 'hydro': 'Thủy', 'solar': 'trời trang', 'wind': 'gió'}
    disp_frames = []
    for label, pat in patterns.items():
        sub = disp[disp['resource_type'].str.contains(pat, case=False, na=False)].set_index('date')
        renamed = sub[['installed_capacity_mw', 'expected_midday_low_load_mw', 'expected_evening_peak_mw']]
        renamed.columns = [f'disp_{label}_installed_mw', f'disp_{label}_midday_mw', f'disp_{label}_evening_mw']
        disp_frames.append(renamed)
    disp_pivot = pd.concat(disp_frames, axis=1)
    disp_pivot.index = pd.to_datetime(disp_pivot.index)
    dates_ts = pd.to_datetime(df.index.date)
    for col in disp_pivot.columns:
        df[col] = dates_ts.map(disp_pivot[col].to_dict()).values
    del disp, disp_pivot; gc.collect()

    weather = pd.read_csv(EXOG_DIR / 'weather_3_regions_30min_open_meteo.csv')
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    weather = weather.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    drop_cols = [c for c in ('date', 'cycle_id', 'source', 'timezone') if c in weather.columns]
    weather = weather.drop(columns=drop_cols)
    weather_cols = [c for c in weather.columns if weather[c].dtype in (np.float64, np.int64, float, int)]
    for col in weather_cols:
        df[col] = weather[col]
    del weather; gc.collect()

    fuel = pd.read_csv(EXOG_DIR / 'fuel_macro_yfinance_clean.csv')
    fuel['date'] = pd.to_datetime(fuel['date'])
    fuel = fuel.sort_values('date').drop_duplicates('date').set_index('date')
    fuel_cols = [c for c in ['coal_proxy_price', 'brent_price', 'gas_proxy_price', 'usd_vnd', 'dxy_index'] if c in fuel.columns]
    for col in fuel_cols:
        df[col] = dates_ts.map(fuel[col].to_dict()).values
    del fuel; gc.collect()

    cal = pd.read_csv(EXOG_DIR / 'calendar_vietnam.csv')
    cal['date'] = pd.to_datetime(cal['date'])
    cal = cal.set_index('date')
    for col in ['is_weekend', 'is_workday', 'is_holiday', 'is_tet', 'is_pre_holiday', 'is_post_holiday', 'season']:
        if col in cal.columns:
            df[col] = dates_ts.map(cal[col].to_dict()).values
    del cal; gc.collect()

    for col in df.select_dtypes(include=[np.number]).columns:
        if not df[col].isna().any(): continue
        if df[col].isna().any(): df[col] = df[col].fillna(df[col].shift(336))
        if df[col].isna().any(): df[col] = df[col].ffill()
        if df[col].isna().any(): df[col] = df[col].bfill()

    print(f"Preprocessing complete. Shape: {df.shape}")
    return df

