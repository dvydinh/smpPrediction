import numpy as np
import pandas as pd


CLEAN_LOWER = 500.0
CLEAN_UPPER = 2500.0
PRICE_LOWER = 0.0
PRICE_UPPER = 1778.6


def clean_mask(values):
    values = np.asarray(values, dtype=float)
    return (values > CLEAN_LOWER) & (values < CLEAN_UPPER)


def target_metrics(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = np.abs(actual - prediction)
    valid = clean_mask(actual)
    clean_error = error[valid]
    denominator = np.abs(actual).sum()
    clean_denominator = np.abs(actual[valid]).sum()
    return {
        "samples": int(len(actual)),
        "mae": float(error.mean()),
        "wmape": float(100.0 * error.sum() / denominator),
        "clean_samples": int(valid.sum()),
        "clean_mae": float(clean_error.mean()),
        "clean_wmape": float(
            100.0 * clean_error.sum() / clean_denominator
        ),
    }


def online_component_average(
    actual,
    index,
    component_predictions,
    lookback_days=28,
):
    index = pd.DatetimeIndex(index)
    actual = np.asarray(actual, dtype=float)
    predictions = {
        name: np.asarray(values, dtype=float)
        for name, values in component_predictions.items()
    }
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    output = np.empty(len(index), dtype=float)
    dates = index.normalize()

    for target_date in dates.unique().sort_values():
        target = dates == target_date
        history = (
            (dates < target_date)
            & (dates >= target_date - pd.Timedelta(days=lookback_days))
            & clean_mask(actual)
        )
        if history.sum() < 48:
            weights = np.full(len(names), 1.0 / len(names))
        else:
            errors = np.mean(
                np.abs(actual[history, None] - matrix[history]),
                axis=0,
            )
            inverse = 1.0 / np.maximum(errors, 25.0)
            weights = inverse / inverse.sum()
        output[target] = matrix[target] @ weights
    return output


def apply_online_adjustment(
    base_prediction,
    index,
    actual,
    history_actual=None,
    bias_days=14,
    cap_days=28,
    cap_ratio=0.95,
    cap_quantile=0.98,
):
    index = pd.DatetimeIndex(index)
    base = np.asarray(base_prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    dates = index.normalize()
    output = base.copy()

    known_actual = pd.Series(dtype=float)
    if history_actual is not None:
        known_actual = pd.Series(history_actual, dtype=float).sort_index()
        known_actual.index = pd.DatetimeIndex(known_actual.index)

    if bias_days:
        residual = pd.Series(actual - base, index=index, dtype=float)
        residual[~clean_mask(actual)] = np.nan
        daily_residual = residual.groupby(dates).median()
        daily_bias = daily_residual.rolling(
            int(bias_days),
            min_periods=3,
        ).median().shift(1)
        output += pd.Series(dates, index=index).map(daily_bias).fillna(0).to_numpy()

    if cap_days and cap_ratio is not None:
        revealed = pd.Series(actual, index=index, dtype=float)
        cap_history = pd.concat([known_actual, revealed])
        cap_history = cap_history[~cap_history.index.duplicated(keep="last")]
        cap_history = cap_history.sort_index()
        rolling_cap = cap_history.rolling(
            f"{int(cap_days)}D",
            min_periods=48,
            closed="left",
        ).quantile(cap_quantile)
        cap_by_date = rolling_cap.reindex(dates.unique())
        row_cap = pd.Series(dates, index=index).map(cap_by_date).to_numpy(dtype=float)
        snap = np.isfinite(row_cap) & (output >= float(cap_ratio) * row_cap)
        output[snap] = row_cap[snap]

    return np.clip(output, PRICE_LOWER, PRICE_UPPER)


def apply_single_day_adjustment(
    base_prediction,
    target_index,
    history_actual,
    history_base_prediction,
    bias_days=14,
    cap_days=28,
    cap_ratio=0.95,
    cap_quantile=0.98,
):
    target_index = pd.DatetimeIndex(target_index)
    if len(target_index) != 48 or target_index.normalize().nunique() != 1:
        raise ValueError("Day-ahead adjustment requires exactly 48 target cycles")

    actual = pd.Series(history_actual, dtype=float).sort_index()
    base_history = pd.Series(history_base_prediction, dtype=float).sort_index()
    common = actual.index.intersection(base_history.index)
    actual = actual.loc[common]
    base_history = base_history.loc[common]
    target_date = target_index[0].normalize()
    adjusted = np.asarray(base_prediction, dtype=float).copy()

    if bias_days:
        recent = (
            (common < target_date)
            & (common >= target_date - pd.Timedelta(days=int(bias_days)))
        )
        valid = recent & clean_mask(actual.to_numpy())
        recent_dates = common[valid].normalize().nunique()
        if recent_dates >= 3:
            adjusted += float(
                np.median(
                    actual.to_numpy()[valid] - base_history.to_numpy()[valid]
                )
            )

    if cap_days and cap_ratio is not None:
        recent = actual[
            (actual.index < target_date)
            & (
                actual.index
                >= target_date - pd.Timedelta(days=int(cap_days))
            )
        ].dropna()
        if len(recent) >= 48:
            cap_value = float(recent.quantile(cap_quantile))
            snap = adjusted >= float(cap_ratio) * cap_value
            adjusted[snap] = cap_value

    return np.clip(adjusted, PRICE_LOWER, PRICE_UPPER)
