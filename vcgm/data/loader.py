from __future__ import annotations
import glob
import logging
from pathlib import Path
import pandas as pd
from vcgm import config as cfg

logger = logging.getLogger(__name__)


def load_smp(path: Path = cfg.SMP_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="first")
    df = df.set_index("datetime")
    logger.info("SMP: %d rows  [%s → %s]", len(df), df.index.min(), df.index.max())
    return df


def load_load(path: Path = cfg.LOAD_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="first")
    df = df.set_index("datetime")
    logger.info("Load: %d rows", len(df))
    return df


def load_dispatch(path: Path = cfg.DISPATCH_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Dispatch: %d rows, %d dates", len(df), df["date"].nunique())
    return df


def get_hydro_files(data_dir: Path = cfg.HYDRO_DIR) -> list[Path]:
    pattern = str(data_dir / "hydro_hourly_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(str(cfg.DATA_DIR / "**" / "hydro_hourly_*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No hydro files in {data_dir}")
    logger.info("Found %d hydro files", len(files))
    return [Path(f) for f in files]


def load_weather(path: Path = cfg.WEATHER_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime", keep="first")
    df = df.set_index("datetime")
    drop = [c for c in ("date", "cycle_id", "source", "timezone") if c in df.columns]
    df = df.drop(columns=drop)
    logger.info("Weather: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_fuel(path: Path = cfg.FUEL_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    df = df.set_index("date")
    logger.info("Fuel: %d rows", len(df))
    return df


def load_calendar(path: Path = cfg.CALENDAR_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    logger.info("Calendar: %d rows", len(df))
    return df
