import numpy as np
import pandas as pd

def add_engineered_features(df):
    CYCLES_PER_DAY = 48
    LAG_SHIFT = 48

    hour_frac = df.index.hour + df.index.minute / 60.0
    df['sin_hour']  = np.sin(2 * np.pi * hour_frac / 24)
    df['cos_hour']  = np.cos(2 * np.pi * hour_frac / 24)
    df['sin_dow']   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['cos_dow']   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)

    for lag in [48, 49, 50, 96, 336]:
        df[f'smp_lag_{lag}']  = df['smp_system_price'].shift(lag)
        df[f'load_lag_{lag}'] = df['load_total_mw'].shift(lag)

    for offset in range(1, 17):
        df[f'smp_recent_{offset}'] = df['smp_system_price'].shift(offset)

    for lag in [48, 336]:
        df[f'smp_north_lag_{lag}'] = df['smp_north_price'].shift(lag)
        df[f'smp_south_lag_{lag}'] = df['smp_south_price'].shift(lag)

    smp_shifted  = df['smp_system_price'].shift(LAG_SHIFT)
    load_shifted = df['load_total_mw'].shift(LAG_SHIFT)

    for label, window in {'24h': 48, '72h': 144}.items():
        df[f'smp_rolling_mean_{label}']  = smp_shifted.rolling(window, min_periods=window//2).mean()
        df[f'smp_rolling_std_{label}']   = smp_shifted.rolling(window, min_periods=window//2).std()
        df[f'load_rolling_mean_{label}'] = load_shifted.rolling(window, min_periods=window//2).mean()
        df[f'load_rolling_std_{label}']  = load_shifted.rolling(window, min_periods=window//2).std()

    df['price_spread_ns_lag48'] = df['smp_north_price'].shift(LAG_SHIFT) - df['smp_south_price'].shift(LAG_SHIFT)
    df['load_ramp_lag48'] = df['load_total_mw'].shift(LAG_SHIFT) - df['load_total_mw'].shift(LAG_SHIFT + 1)
    df['smp_yesterday_mean']  = smp_shifted.rolling(CYCLES_PER_DAY).mean()
    df['smp_yesterday_max']   = smp_shifted.rolling(CYCLES_PER_DAY).max()
    df['smp_yesterday_min']   = smp_shifted.rolling(CYCLES_PER_DAY).min()
    df['smp_yesterday_zero_ratio'] = (df['smp_system_price'].shift(LAG_SHIFT) <= 2.0).astype(float).rolling(CYCLES_PER_DAY).mean()

    if 'hydro_total_discharge_m3s' in df.columns:
        df['hydro_discharge_rolling_24h'] = df['hydro_total_discharge_m3s'].shift(LAG_SHIFT).rolling(48, min_periods=24).mean()

    df = df.drop(columns=['smp_north_price', 'smp_central_price', 'smp_south_price'], errors='ignore')

    if 'shortwave_radiation_hcmc' in df.columns and 'disp_solar_installed_mw' in df.columns:
        df['solar_gen_proxy'] = (df['shortwave_radiation_hcmc'] / 1000.0) * df['disp_solar_installed_mw'] * 0.75
    else:
        df['solar_gen_proxy'] = 0

    if 'wind_speed_hcmc' in df.columns and 'disp_wind_installed_mw' in df.columns:
        df['wind_gen_proxy'] = (df['wind_speed_hcmc'] / 10.0) * df['disp_wind_installed_mw'] * 0.3
    else:
        df['wind_gen_proxy'] = 0

    if 'load_total_mw' in df.columns and 'solar_gen_proxy' in df.columns:
        df['residual_load_proxy'] = df['load_total_mw'].shift(LAG_SHIFT) - df['solar_gen_proxy'].shift(LAG_SHIFT) - df['wind_gen_proxy'].shift(LAG_SHIFT)

    if 'disp_total_installed_mw' in df.columns and 'residual_load_proxy' in df.columns:
        hydro_cap = df['disp_hydro_installed_mw'].fillna(0) if 'disp_hydro_installed_mw' in df.columns else 0
        solar_cap = df['disp_solar_installed_mw'].fillna(0) if 'disp_solar_installed_mw' in df.columns else 0
        wind_cap = df['disp_wind_installed_mw'].fillna(0) if 'disp_wind_installed_mw' in df.columns else 0
        thermal_cap = df['disp_total_installed_mw'] - hydro_cap - solar_cap - wind_cap
        df['thermal_margin_proxy'] = thermal_cap - df['residual_load_proxy']

    if 'hydro_water_level_m' in df.columns:
        df['hydro_depletion_index'] = df['hydro_water_level_m'].shift(LAG_SHIFT)

    if 'thermal_margin_proxy' in df.columns and 'temperature_danang' in df.columns:
        df['margin_temp_interaction'] = df['thermal_margin_proxy'] * df['temperature_danang'].shift(LAG_SHIFT)

    df['is_spike'] = (df['smp_system_price'] >= 1500).astype(int)
    df['is_zero'] = (df['smp_system_price'] <= 100).astype(int)
    df['spike_prob_24h'] = df['is_spike'].shift(LAG_SHIFT).rolling(48).mean()
    df['spike_prob_72h'] = df['is_spike'].shift(LAG_SHIFT).rolling(144).mean()
    df['zero_prob_24h'] = df['is_zero'].shift(LAG_SHIFT).rolling(48).mean()
    df['zero_prob_72h'] = df['is_zero'].shift(LAG_SHIFT).rolling(144).mean()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any(): df[col] = df[col].ffill().bfill()
        
    print(f"[INFO] FE_final_shape: {df.shape}, NaN count: {df.isna().sum().sum()}")
    return df

