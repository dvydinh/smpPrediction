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
