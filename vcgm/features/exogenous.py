from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from vcgm import config as cfg

logger = logging.getLogger(__name__)


def process_hydro(hydro_files: list[Path]) -> pd.DataFrame:
    chunks = []
    for f in hydro_files:
        df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df["datetime"])
        agg_chunk = df.groupby("datetime").agg({
            "inflow_m3s": "sum",
            "total_discharge_m3s": "sum",
            "plant_discharge_m3s": "sum",
            "spill_discharge_m3s": "sum",
            "water_level_m": "mean",
        })
        chunks.append(agg_chunk)

    agg = pd.concat(chunks).groupby(level=0).mean()
    agg.columns = ["hydro_" + c for c in agg.columns]
    agg = agg.sort_index()

    idx_30 = pd.date_range(agg.index.min(), agg.index.max() + pd.Timedelta("30min"), freq="30min")
    agg = agg.reindex(idx_30).ffill()
    logger.info("Hydro: %d rows (30-min, ffill from hourly)", len(agg))
    return agg


def process_dispatch(dispatch_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, pat in cfg.DISPATCH_RESOURCE_PATTERNS.items():
        sub = dispatch_df[dispatch_df["resource_type"].str.contains(pat, case=False, na=False)]
        sub = sub.set_index("date")
        renamed = sub[["installed_capacity_mw", "expected_midday_low_load_mw", "expected_evening_peak_mw"]]
        renamed.columns = [f"disp_{label}_installed_mw", f"disp_{label}_midday_mw", f"disp_{label}_evening_mw"]
        frames.append(renamed)
    result = pd.concat(frames, axis=1)
    result.index = pd.to_datetime(result.index)
    logger.info("Dispatch: %d dates, %d cols", len(result), len(result.columns))
    return result


def process_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    numeric = weather_df.select_dtypes(include=[np.number])
    logger.info("Weather: %d cols retained", len(numeric.columns))
    return numeric


def process_fuel(fuel_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in cfg.FUEL_COLS if c in fuel_df.columns]
    result = fuel_df[cols].copy()
    logger.info("Fuel: %d cols", len(result.columns))
    return result
