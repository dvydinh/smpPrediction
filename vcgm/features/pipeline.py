from __future__ import annotations
import gc
import logging
import numpy as np
import pandas as pd
from vcgm import config as cfg
from vcgm.data import loader, alignment
from vcgm.features import temporal, lagged, exogenous

logger = logging.getLogger(__name__)


def build_design_matrix(save=True):
    master_idx = alignment.build_master_timeline()
    df = pd.DataFrame(index=master_idx)
    df.index.name = "datetime"

    smp = loader.load_smp()
    for col in ["smp_system_price", "smp_north_price", "smp_central_price", "smp_south_price"]:
        df[col] = smp[col]
    df["cycle_id"] = df.index.hour * 2 + df.index.minute // 30

    load = loader.load_load()
    for col in ["load_total_mw", "load_north_mw", "load_central_mw", "load_south_mw"]:
        df[col] = load[col]

    hydro_files = loader.get_hydro_files()
    hydro = exogenous.process_hydro(hydro_files)
    for col in hydro.columns:
        df[col] = hydro[col]
    del hydro; gc.collect()

    disp_raw = loader.load_dispatch()
    disp = exogenous.process_dispatch(disp_raw)
    del disp_raw
    dates_ts = pd.to_datetime(df.index.date)
    for col in disp.columns:
        df[col] = dates_ts.map(disp[col].to_dict()).values
    del disp; gc.collect()

    weather_raw = loader.load_weather()
    weather = exogenous.process_weather(weather_raw)
    del weather_raw
    for col in weather.columns:
        df[col] = weather[col]
    del weather; gc.collect()

    fuel_raw = loader.load_fuel()
    fuel = exogenous.process_fuel(fuel_raw)
    del fuel_raw
    for col in fuel.columns:
        df[col] = dates_ts.map(fuel[col].to_dict()).values
    del fuel; gc.collect()

    cal = loader.load_calendar()
    df = temporal.add_calendar_features(df, cal)
    del cal; gc.collect()

    logger.info("Running imputation …")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if not df[col].isna().any():
            continue
        df[col] = df[col].interpolate(method="linear", limit=2)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].shift(336))
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    logger.info("Engineering features …")
    df = temporal.add_cyclical_features(df)
    df = lagged.add_smp_lags(df)
    df = lagged.add_rolling_stats(df)
    df = lagged.add_derived_features(df)
    df = lagged.add_physics_informed_features(df)

    if "hydro_total_discharge_m3s" in df.columns:
        hydro_shifted = df["hydro_total_discharge_m3s"].shift(cfg.LAG_SAFETY_SHIFT)
        df["hydro_discharge_rolling_24h"] = hydro_shifted.rolling(48, min_periods=24).mean()

    df = df.drop(columns=["smp_north_price", "smp_central_price", "smp_south_price"], errors="ignore")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    logger.info("Design matrix: %s", df.shape)

    if save:
        df.to_parquet(cfg.DESIGN_MATRIX_FILE)
        logger.info("Saved to %s", cfg.DESIGN_MATRIX_FILE)

    return df


def build_daily_matrices(df, feature_cols, target="smp_system_price"):
    df = df.copy()
    df["_date"] = pd.to_datetime(df.index.date)
    dates = sorted(df["_date"].unique())

    X_list, Y_list, date_list = [], [], []

    for i in range(len(dates) - 1):
        day_d, day_d1 = dates[i], dates[i + 1]
        snapshot = pd.Timestamp(day_d) + pd.Timedelta(hours=7, minutes=30)
        if snapshot not in df.index:
            continue

        feat_row = df.loc[snapshot, feature_cols]
        d1_start = pd.Timestamp(day_d1)
        targets = df.loc[d1_start:d1_start + pd.Timedelta(hours=23, minutes=30), target]

        if len(targets) != cfg.CYCLES_PER_DAY:
            continue
        if feat_row.isna().all() or targets.isna().any():
            continue

        X_list.append(feat_row.values)
        Y_list.append(targets.values)
        date_list.append(day_d1)

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)
    dates_arr = np.array(date_list)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = medians[j]
        logger.warning("Imputed %d NaN in feature matrix", nan_mask.sum())

    logger.info("Daily matrices: X%s  Y%s  dates %s→%s", X.shape, Y.shape, dates_arr[0], dates_arr[-1])
    return X, Y, dates_arr
