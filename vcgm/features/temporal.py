from __future__ import annotations
import numpy as np
import pandas as pd
from vcgm import config as cfg


def add_cyclical_features(df):
    hour = df.index.hour + df.index.minute / 60.0
    df["sin_hour"]  = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * hour / 24)
    df["sin_dow"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["sin_month"] = np.sin(2 * np.pi * df.index.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df.index.month / 12)
    return df


def add_calendar_features(df, calendar_df):
    dates_ts = pd.to_datetime(df.index.date)
    for col in cfg.CALENDAR_COLS:
        if col in calendar_df.columns:
            df[col] = dates_ts.map(calendar_df[col].to_dict()).values
    return df
