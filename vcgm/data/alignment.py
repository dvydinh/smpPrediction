from __future__ import annotations
import logging
import pandas as pd
from vcgm import config as cfg

logger = logging.getLogger(__name__)


def build_master_timeline(start=cfg.TIMELINE_START, end=None, freq=cfg.TIMELINE_FREQ):
    if end is None:
        end = "2026-06-19 23:30:00"
    idx = pd.date_range(start=start, end=end, freq=freq)
    logger.info("Timeline: %s → %s  (%d intervals)", idx[0], idx[-1], len(idx))
    return idx


def align_to_timeline(df, master_idx, columns=None):
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    aligned = df.reindex(master_idx)
    n_missing = aligned.iloc[:, 0].isna().sum() if len(aligned.columns) > 0 else 0
    logger.info("Aligned %d → %d slots (missing: %d)", len(df), len(aligned), n_missing)
    return aligned
