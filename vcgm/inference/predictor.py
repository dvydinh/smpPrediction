from __future__ import annotations
import logging
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from vcgm import config as cfg
from vcgm.data import loader
from vcgm.features import temporal, lagged, exogenous
from vcgm.models.direct_forecast import DirectForecaster

logger = logging.getLogger(__name__)


class SmpPredictor:

    def __init__(self, model_dir=cfg.MODEL_DIR):
        self.forecaster = DirectForecaster.load(model_dir)
        self.feature_names = self.forecaster.feature_names

    def predict(self, current_datetime=None):
        t0 = time.time()
        if current_datetime is None:
            current_datetime = pd.Timestamp.now()
        else:
            current_datetime = pd.Timestamp(current_datetime)

        day_d = current_datetime.normalize()
        target_date = (day_d + pd.Timedelta(days=1)).date()
        target_start = pd.Timestamp(target_date)
        warnings_list = []
        status = "success"

        try:
            smp = loader.load_smp()
            smp = smp[smp.index < current_datetime]
        except Exception as e:
            warnings_list.append(f"SMP: {e}"); smp = None; status = "partial"

        try:
            load_df = loader.load_load()
            load_df = load_df[load_df.index < current_datetime]
        except Exception as e:
            warnings_list.append(f"Load: {e}"); load_df = None; status = "partial"

        try:
            weather_raw = loader.load_weather()
            weather = exogenous.process_weather(weather_raw)
            weather_target = weather.loc[target_start:target_start + pd.Timedelta("23h30min")]
            weather_mean = weather_target.mean() if len(weather_target) == cfg.CYCLES_PER_DAY else pd.Series(dtype=float)
        except Exception as e:
            warnings_list.append(f"Weather: {e}"); weather_mean = pd.Series(dtype=float); status = "partial"

        try:
            hydro_files = loader.get_hydro_files()
            hydro = exogenous.process_hydro(hydro_files)
            hydro = hydro[hydro.index < current_datetime]
        except Exception as e:
            warnings_list.append(f"Hydro: {e}"); hydro = None; status = "partial"

        try:
            disp_raw = loader.load_dispatch()
            disp = exogenous.process_dispatch(disp_raw)
        except Exception as e:
            warnings_list.append(f"Dispatch: {e}"); disp = None; status = "partial"

        try:
            fuel_raw = loader.load_fuel()
            fuel = exogenous.process_fuel(fuel_raw)
            fuel = fuel[fuel.index < current_datetime]
        except Exception as e:
            warnings_list.append(f"Fuel: {e}"); fuel = None; status = "partial"

        try:
            cal = loader.load_calendar()
        except Exception as e:
            warnings_list.append(f"Calendar: {e}"); cal = None; status = "partial"

        snapshot_time = day_d + pd.Timedelta(hours=7, minutes=30)
        feats = {"cycle_id": 15}

        hour_frac = snapshot_time.hour + snapshot_time.minute / 60.0
        feats["sin_hour"] = np.sin(2 * np.pi * hour_frac / 24)
        feats["cos_hour"] = np.cos(2 * np.pi * hour_frac / 24)
        dow = snapshot_time.dayofweek
        feats["sin_dow"] = np.sin(2 * np.pi * dow / 7)
        feats["cos_dow"] = np.cos(2 * np.pi * dow / 7)
        month = snapshot_time.month
        feats["sin_month"] = np.sin(2 * np.pi * month / 12)
        feats["cos_month"] = np.cos(2 * np.pi * month / 12)

        if smp is not None:
            s_smp = smp["smp_system_price"]
            for lag in cfg.SMP_LAGS:
                t = snapshot_time - pd.Timedelta(minutes=30 * lag)
                idx = s_smp.index.get_indexer([t], method="pad")[0]
                feats[f"smp_lag_{lag}"] = s_smp.iloc[idx] if idx >= 0 else 0

            recent = s_smp.loc[:snapshot_time - pd.Timedelta("24h")]
            if len(recent) >= 48:
                feats["smp_rolling_mean_24h"] = recent.iloc[-48:].mean()
                feats["smp_rolling_std_24h"] = recent.iloc[-48:].std()
            if len(recent) >= 144:
                feats["smp_rolling_mean_72h"] = recent.iloc[-144:].mean()
                feats["smp_rolling_std_72h"] = recent.iloc[-144:].std()

            t48 = snapshot_time - pd.Timedelta("24h")
            idx = smp.index.get_indexer([t48], method="pad")[0]
            if idx >= 0:
                feats["price_spread_ns_lag48"] = smp["smp_north_price"].iloc[idx] - smp["smp_south_price"].iloc[idx]

            yest = s_smp.loc[snapshot_time - pd.Timedelta("48h"):snapshot_time - pd.Timedelta("24h")]
            if len(yest) > 0:
                feats["smp_yesterday_mean"] = yest.mean()
                feats["smp_yesterday_max"] = yest.max()
                feats["smp_yesterday_min"] = yest.min()
                feats["smp_yesterday_zero_ratio"] = (yest <= cfg.NEAR_ZERO_THRESHOLD).mean()

        if load_df is not None:
            s_load = load_df["load_total_mw"]
            for lag in cfg.LOAD_LAGS:
                t = snapshot_time - pd.Timedelta(minutes=30 * lag)
                idx = s_load.index.get_indexer([t], method="pad")[0]
                feats[f"load_lag_{lag}"] = s_load.iloc[idx] if idx >= 0 else 0
            recent = s_load.loc[:snapshot_time - pd.Timedelta("24h")]
            if len(recent) >= 48:
                feats["load_rolling_mean_24h"] = recent.iloc[-48:].mean()
                feats["load_rolling_std_24h"] = recent.iloc[-48:].std()
            if len(recent) >= 144:
                feats["load_rolling_mean_72h"] = recent.iloc[-144:].mean()
                feats["load_rolling_std_72h"] = recent.iloc[-144:].std()
            t48 = snapshot_time - pd.Timedelta("24h")
            t49 = snapshot_time - pd.Timedelta("24h30min")
            i48 = s_load.index.get_indexer([t48], method="pad")[0]
            i49 = s_load.index.get_indexer([t49], method="pad")[0]
            if i48 >= 0 and i49 >= 0:
                feats["load_ramp_lag48"] = s_load.iloc[i48] - s_load.iloc[i49]

        if hydro is not None:
            t24 = snapshot_time - pd.Timedelta("24h")
            idx = hydro.index.get_indexer([t24], method="pad")[0]
            if idx >= 0:
                for col in hydro.columns:
                    feats[col] = hydro[col].iloc[idx]
            recent = hydro.loc[:t24, "hydro_total_discharge_m3s"]
            if len(recent) >= 48:
                feats["hydro_discharge_rolling_24h"] = recent.iloc[-48:].mean()

        if disp is not None:
            idx = disp.index.get_indexer([day_d], method="pad")[0]
            if idx >= 0:
                for col in disp.columns:
                    feats[col] = disp[col].iloc[idx]

        if fuel is not None and len(fuel) > 0:
            for col in fuel.columns:
                feats[col] = fuel[col].iloc[-1]

        for col in weather_mean.index:
            feats[col] = weather_mean[col]

        if cal is not None:
            idx = cal.index.get_indexer([pd.Timestamp(target_date)], method="pad")[0]
            if idx >= 0:
                for col in cfg.CALENDAR_COLS:
                    if col in cal.columns:
                        feats[col] = cal[col].iloc[idx]

        X = np.zeros((1, len(self.feature_names)))
        for i, col in enumerate(self.feature_names):
            X[0, i] = feats.get(col, 0.0)

        try:
            preds = self.forecaster.predict(X)[0]
        except Exception as e:
            warnings_list.append(f"Prediction: {e}")
            status = "failed"
            preds = np.full(48, np.nan)

        elapsed = time.time() - t0
        return {
            "predictions": preds.tolist(),
            "target_date": str(target_date),
            "cycles": [f"{c // 2:02d}:{(c % 2) * 30:02d}" for c in range(48)],
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "warnings": warnings_list,
        }
