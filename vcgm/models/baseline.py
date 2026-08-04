from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def naive_lag7(Y_daily):
    preds = np.full_like(Y_daily, np.nan)
    for i in range(7, len(Y_daily)):
        preds[i] = Y_daily[i - 7]

    valid = ~np.isnan(preds).any(axis=1)
    y_true = Y_daily[valid].flatten()
    y_pred = preds[valid].flatten()

    return preds, {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
