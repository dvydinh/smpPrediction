from __future__ import annotations
import pandas as pd
from vcgm import config as cfg


def add_smp_lags(df):
    for lag in cfg.SMP_LAGS:
        df[f"smp_lag_{lag}"] = df["smp_system_price"].shift(lag)
    for lag in cfg.REGIONAL_LAGS:
        if "smp_north_price" in df.columns:
            df[f"smp_north_lag_{lag}"] = df["smp_north_price"].shift(lag)
            df[f"smp_south_lag_{lag}"] = df["smp_south_price"].shift(lag)
    for lag in cfg.LOAD_LAGS:
        df[f"load_lag_{lag}"] = df["load_total_mw"].shift(lag)
    return df


def add_rolling_stats(df):
    smp_shifted  = df["smp_system_price"].shift(cfg.LAG_SAFETY_SHIFT)
    load_shifted = df["load_total_mw"].shift(cfg.LAG_SAFETY_SHIFT)

    for label, w in cfg.ROLLING_WINDOWS.items():
        df[f"smp_rolling_mean_{label}"]  = smp_shifted.rolling(w, min_periods=w // 2).mean()
        df[f"smp_rolling_std_{label}"]   = smp_shifted.rolling(w, min_periods=w // 2).std()
        df[f"load_rolling_mean_{label}"] = load_shifted.rolling(w, min_periods=w // 2).mean()
        df[f"load_rolling_std_{label}"]  = load_shifted.rolling(w, min_periods=w // 2).std()
    return df


def add_derived_features(df):
    shift = cfg.LAG_SAFETY_SHIFT
    if "smp_north_price" in df.columns:
        df["price_spread_ns_lag48"] = df["smp_north_price"].shift(shift) - df["smp_south_price"].shift(shift)
    df["load_ramp_lag48"] = df["load_total_mw"].shift(shift) - df["load_total_mw"].shift(shift + 1)

    smp_shifted = df["smp_system_price"].shift(shift)
    df["smp_yesterday_mean"] = smp_shifted.rolling(cfg.CYCLES_PER_DAY).mean()
    df["smp_yesterday_max"]  = smp_shifted.rolling(cfg.CYCLES_PER_DAY).max()
    df["smp_yesterday_min"]  = smp_shifted.rolling(cfg.CYCLES_PER_DAY).min()
    df["smp_yesterday_zero_ratio"] = (
        (df["smp_system_price"].shift(shift) <= cfg.NEAR_ZERO_THRESHOLD)
        .astype(float).rolling(cfg.CYCLES_PER_DAY).mean()
    )
    return df


def add_physics_informed_features(df):
    shift = cfg.LAG_SAFETY_SHIFT
    if "shortwave_radiation_hcmc" in df.columns and "disp_solar_installed_mw" in df.columns:
        df["solar_gen_proxy"] = (df["shortwave_radiation_hcmc"] / 1000.0) * df["disp_solar_installed_mw"] * 0.75
    else:
        df["solar_gen_proxy"] = 0

    if "wind_speed_hcmc" in df.columns and "disp_wind_installed_mw" in df.columns:
        df["wind_gen_proxy"] = (df["wind_speed_hcmc"] / 10.0) * df["disp_wind_installed_mw"] * 0.3
    else:
        df["wind_gen_proxy"] = 0

    if "load_total_mw" in df.columns and "solar_gen_proxy" in df.columns:
        df["residual_load_proxy"] = df["load_total_mw"].shift(shift) - df["solar_gen_proxy"].shift(shift) - df["wind_gen_proxy"].shift(shift)

    if "disp_total_installed_mw" in df.columns and "residual_load_proxy" in df.columns:
        hydro_cap = df["disp_hydro_installed_mw"].fillna(0) if "disp_hydro_installed_mw" in df.columns else 0
        solar_cap = df["disp_solar_installed_mw"].fillna(0) if "disp_solar_installed_mw" in df.columns else 0
        wind_cap = df["disp_wind_installed_mw"].fillna(0) if "disp_wind_installed_mw" in df.columns else 0
        thermal_cap = df["disp_total_installed_mw"] - hydro_cap - solar_cap - wind_cap
        df["thermal_margin_proxy"] = thermal_cap - df["residual_load_proxy"]

    if "hydro_water_level_m" in df.columns:
        df["hydro_depletion_index"] = df["hydro_water_level_m"].shift(shift)

    return df
