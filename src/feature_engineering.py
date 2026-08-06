import numpy as np
import pandas as pd

def add_engineered_features(df):
    """
    Safely engineers features for the Day-Ahead Market SMP Prediction.
    Strictly simulates the 08:00 Day D operational blindspot:
    To predict Day D+1 (target day), the last available data is 07:30 Day D.
    """
    df = df.copy()
    
    # Time based features (perfectly known in advance)
    hour_frac = df.index.hour + df.index.minute / 60.0
    df['sin_hour']  = np.sin(2 * np.pi * hour_frac / 24)
    df['cos_hour']  = np.cos(2 * np.pi * hour_frac / 24)
    df['sin_dow']   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['cos_dow']   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # 0 is 00:00, 15 is 07:30, 47 is 23:30
    cycle_id = df.index.hour * 2 + (df.index.minute == 30).astype(int)
    
    # =========================================================
    # 1. SAME-CYCLE LAGS (Strictly 08:00 Blindspot Safe)
    # =========================================================
    # For Day D+1 cycle `c`:
    # If c <= 15 (00:00 to 07:30): Day D cycle `c` IS available. We can use shift(48).
    # If c > 15 (08:00 to 23:30): Day D cycle `c` IS NOT available. We must use Day D-1 cycle `c`, which is shift(96).
    
    def safe_same_cycle_lag(col_name, new_col_prefix):
        if col_name not in df.columns: return
        df[f'{new_col_prefix}_1d'] = np.where(cycle_id <= 15, df[col_name].shift(48), df[col_name].shift(96))
        df[f'{new_col_prefix}_2d'] = df[col_name].shift(96)
        df[f'{new_col_prefix}_7d'] = df[col_name].shift(336)
        
    safe_same_cycle_lag('smp_system_price', 'smp_same_cycle')
    safe_same_cycle_lag('load_total_mw', 'load_same_cycle')
    safe_same_cycle_lag('smp_north_price', 'smp_north_same_cycle')
    safe_same_cycle_lag('smp_south_price', 'smp_south_same_cycle')

    # =========================================================
    # 2. MORNING AGGREGATES (Day D 00:00 to 07:30)
    # =========================================================
    # This block is fully available at 08:00 Day D.
    df['date_str'] = df.index.date.astype(str)
    
    morning_mask = (cycle_id <= 15)
    morning_stats = df[morning_mask].groupby('date_str').agg(
        morning_smp_mean=('smp_system_price', 'mean'),
        morning_smp_max=('smp_system_price', 'max'),
        morning_smp_min=('smp_system_price', 'min'),
        morning_load_mean=('load_total_mw', 'mean')
    )
    # Shift by 1 day because Day D+1 uses Day D's morning stats
    morning_stats = morning_stats.shift(1)
    df = df.join(morning_stats, on='date_str')

    # =========================================================
    # 3. FULL DAY AGGREGATES (Day D-1)
    # =========================================================
    # Day D-1 is the most recent FULL day available.
    daily_stats = df.groupby('date_str').agg(
        prev_full_smp_mean=('smp_system_price', 'mean'),
        prev_full_smp_max=('smp_system_price', 'max'),
        prev_full_smp_min=('smp_system_price', 'min'),
        prev_full_gate_prob=('smp_system_price', lambda x: (x <= 500).mean())
    )
    # Shift by 2 days because Day D+1 uses Day D-1's full daily stats
    daily_stats = daily_stats.shift(2)
    df = df.join(daily_stats, on='date_str')
    
    df = df.drop(columns=['date_str'])

    # =========================================================
    # 4. WEATHER & PROXIES (Available for Day D+1)
    # =========================================================
    if 'shortwave_radiation_hcmc' in df.columns and 'disp_solar_installed_mw' in df.columns:
        df['solar_gen_proxy'] = (df['shortwave_radiation_hcmc'] / 1000.0) * df['disp_solar_installed_mw'] * 0.75
    else:
        df['solar_gen_proxy'] = 0

    if 'wind_speed_hcmc' in df.columns and 'disp_wind_installed_mw' in df.columns:
        df['wind_gen_proxy'] = (df['wind_speed_hcmc'] / 10.0) * df['disp_wind_installed_mw'] * 0.3
    else:
        df['wind_gen_proxy'] = 0

    if 'load_total_mw' in df.columns and 'solar_gen_proxy' in df.columns:
        df['residual_load_proxy'] = df['load_same_cycle_1d'] - df['solar_gen_proxy'] - df['wind_gen_proxy']

    if 'disp_total_installed_mw' in df.columns and 'residual_load_proxy' in df.columns:
        hydro_cap = df['disp_hydro_installed_mw'].fillna(0) if 'disp_hydro_installed_mw' in df.columns else 0
        solar_cap = df['disp_solar_installed_mw'].fillna(0) if 'disp_solar_installed_mw' in df.columns else 0
        wind_cap = df['disp_wind_installed_mw'].fillna(0) if 'disp_wind_installed_mw' in df.columns else 0
        thermal_cap = df['disp_total_installed_mw'] - hydro_cap - solar_cap - wind_cap
        df['thermal_margin_proxy'] = thermal_cap - df['residual_load_proxy']

    df = df.drop(columns=['smp_north_price', 'smp_central_price', 'smp_south_price'], errors='ignore')

    # =========================================================
    # 5. ROLLING VOLATILITY & MOMENTUM (Blindspot Safe)
    # =========================================================
    # All use shift(48) minimum to ensure we only use data available before 08:00 Day D
    if 'smp_same_cycle_1d' in df.columns:
        # Price momentum: how much did the price change vs 2 days ago?
        df['smp_momentum_1d_2d'] = df['smp_same_cycle_1d'] - df['smp_same_cycle_2d']
        
        # Price spread North-South (using lagged values, already blindspot-safe)
        if 'smp_north_same_cycle_1d' in df.columns and 'smp_south_same_cycle_1d' in df.columns:
            df['smp_spread_ns_1d'] = df['smp_north_same_cycle_1d'] - df['smp_south_same_cycle_1d']

    # Rolling volatility: std of SMP over past 48 cycles (1 day), shifted by 48 for safety
    if 'smp_system_price' in df.columns:
        df['smp_rolling_std_1d'] = df['smp_system_price'].shift(48).rolling(48, min_periods=24).std()
        df['smp_rolling_std_7d'] = df['smp_system_price'].shift(48).rolling(336, min_periods=168).std()
        # Rolling mean for mean-reversion signal
        df['smp_rolling_mean_7d'] = df['smp_system_price'].shift(48).rolling(336, min_periods=168).mean()
        # Deviation from 7-day mean (mean reversion signal)
        df['smp_dev_from_7d_mean'] = df['smp_same_cycle_1d'] - df['smp_rolling_mean_7d']

    # Weekend/Holiday flag (binary, complements sin/cos encoding)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any(): 
            df[col] = df[col].ffill().bfill()
        
    print(f"Feature engineering complete. Shape: {df.shape}, NaN count: {df.isna().sum().sum()}")
    return df
