from __future__ import annotations
import numpy as np
import pandas as pd
from vcgm import config as cfg


def profile_series(series, name=""):
    valid = series.dropna()
    return {
        "name": name, "count": len(series),
        "nan": int(series.isna().sum()),
        "nan_pct": round(100.0 * series.isna().sum() / max(len(series), 1), 2),
        "mean": round(float(valid.mean()), 2) if len(valid) else None,
        "std": round(float(valid.std()), 2) if len(valid) else None,
        "min": round(float(valid.min()), 2) if len(valid) else None,
        "median": round(float(valid.median()), 2) if len(valid) else None,
        "max": round(float(valid.max()), 2) if len(valid) else None,
    }


def profile_smp(smp_series):
    valid = smp_series.dropna()
    n = len(valid)
    base = profile_series(smp_series, "smp_system_price")
    base.update({
        "near_zero_le2": int((valid <= cfg.NEAR_ZERO_THRESHOLD).sum()),
        "near_zero_le2_pct": round(100.0 * (valid <= cfg.NEAR_ZERO_THRESHOLD).sum() / n, 2),
        "above_1000": int((valid > 1000).sum()),
        "above_1000_pct": round(100.0 * (valid > 1000).sum() / n, 2),
        "price_cap_hits": int((valid >= cfg.PRICE_CAP_VND - 0.1).sum()),
    })
    return base


def find_gaps(index, mask):
    missing_times = index[mask]
    if len(missing_times) == 0:
        return []
    gaps, start, prev = [], missing_times[0], missing_times[0]
    for t in missing_times[1:]:
        if (t - prev) > pd.Timedelta("30min"):
            gaps.append({"start": str(start), "end": str(prev), "length": int((prev - start) / pd.Timedelta("30min")) + 1})
            start = t
        prev = t
    gaps.append({"start": str(start), "end": str(prev), "length": int((prev - start) / pd.Timedelta("30min")) + 1})
    return gaps
