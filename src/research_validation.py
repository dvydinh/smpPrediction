import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.daily_models import CINGLearForecaster
from src.online_calibration import (
    PRICE_LOWER,
    PRICE_UPPER,
    apply_online_adjustment,
    apply_single_day_adjustment,
    clean_mask,
    online_component_average,
    target_metrics,
)


PRETEST_YEARS = (2024, 2025)
CALIBRATION_WINDOWS = (45, 60, 90, 180, 365, 730, 1460)
ONLINE_WEIGHT_DAYS = 28
POSTPROCESS_CONFIGS = (
    (0, 0, None),
    (7, 28, 0.95),
    (14, 14, 0.95),
    (14, 28, 0.90),
    (14, 28, 0.95),
    (14, 28, 0.98),
    (14, 56, 0.95),
    (28, 28, 0.95),
)


class AdaptiveWindowEnsemble:
    def __init__(
        self,
        selected_candidate="adaptive_median",
        windows=CALIBRATION_WINDOWS,
        bias_days=0,
        cap_days=0,
        cap_ratio=None,
    ):
        self.selected_candidate_ = selected_candidate
        self.windows = tuple(int(value) for value in windows)
        self.bias_days_ = int(bias_days)
        self.cap_days_ = int(cap_days)
        self.cap_ratio_ = None if cap_ratio is None else float(cap_ratio)
        self.models = {}
        self.model_order_ = []
        self.selected_features = []

    def fit(self, X, y):
        frame = X.copy().sort_index()
        frame["smp_system_price"] = np.asarray(y.loc[frame.index], dtype=float)
        forecast_start = frame.index.max() + pd.Timedelta(minutes=30)
        self.models = _fit_window_models(
            frame,
            self.selected_features,
            forecast_start,
            self.windows,
        )
        self.model_order_ = list(self.models)
        if not self.model_order_:
            raise ValueError("No calibration window has enough clean samples")
        self.models["lgb"] = self.models[self.model_order_[-1]]

        if "cing_lear" in self.selected_candidate_ or self.selected_candidate_ == "hybrid_median":
            multitask = CINGLearForecaster()
            multitask.fit(frame[self.selected_features], frame["smp_system_price"])
            self.models["cing_lear"] = multitask
        return self

    def _lgb_predictions(self, X):
        return {
            name: self.models[name].predict(X)
            for name in self.model_order_
        }

    def _base_prediction(self, X, lgb_predictions=None):
        predictions = lgb_predictions or self._lgb_predictions(X)
        if self.selected_candidate_ in predictions:
            return predictions[self.selected_candidate_]
        median = np.median(
            np.column_stack(list(predictions.values())),
            axis=1,
        )
        if self.selected_candidate_ in {"adaptive_median", "online_average"}:
            return median
        if self.selected_candidate_ == "cing_lear":
            return self.models["cing_lear"].predict(X)
        if self.selected_candidate_ == "hybrid_median":
            multitask = self.models["cing_lear"].predict(X)
            return np.median(np.column_stack([median, multitask]), axis=1)
        raise ValueError(f"Unknown adaptive candidate {self.selected_candidate_}")

    def predict(self, X):
        return self._base_prediction(X)

    def predict_walk_forward(self, X, actual, history_actual=None):
        lgb_predictions = self._lgb_predictions(X)
        if self.selected_candidate_ == "online_average":
            base = online_component_average(
                actual,
                X.index,
                lgb_predictions,
                lookback_days=ONLINE_WEIGHT_DAYS,
            )
        else:
            base = self._base_prediction(X, lgb_predictions)
        return apply_online_adjustment(
            base,
            X.index,
            actual,
            history_actual=history_actual,
            bias_days=self.bias_days_,
            cap_days=self.cap_days_,
            cap_ratio=self.cap_ratio_,
        )

    def predict_with_history(self, X, history_X, history_y):
        lgb_target = self._lgb_predictions(X)
        lgb_history = self._lgb_predictions(history_X)
        if self.selected_candidate_ == "online_average":
            target_date = X.index.min().normalize()
            recent = (
                (history_X.index < target_date)
                & (
                    history_X.index
                    >= target_date - pd.Timedelta(days=ONLINE_WEIGHT_DAYS)
                )
                & clean_mask(history_y.to_numpy())
            )
            if recent.sum() < 48:
                weights = np.full(len(self.model_order_), 1.0 / len(self.model_order_))
            else:
                errors = []
                actual = history_y.to_numpy(dtype=float)[recent]
                for name in self.model_order_:
                    errors.append(np.mean(np.abs(actual - lgb_history[name][recent])))
                inverse = 1.0 / np.maximum(np.asarray(errors), 25.0)
                weights = inverse / inverse.sum()
            target_matrix = np.column_stack([
                lgb_target[name] for name in self.model_order_
            ])
            history_matrix = np.column_stack([
                lgb_history[name] for name in self.model_order_
            ])
            base_target = target_matrix @ weights
            base_history = history_matrix @ weights
        else:
            base_target = self._base_prediction(X, lgb_target)
            base_history = self._base_prediction(history_X, lgb_history)

        return apply_single_day_adjustment(
            base_target,
            X.index,
            history_y,
            pd.Series(base_history, index=history_X.index),
            bias_days=self.bias_days_,
            cap_days=self.cap_days_,
            cap_ratio=self.cap_ratio_,
        )

    def save(self, output_dir, model_name="adaptive_window.pkl"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, model_name)
        with open(path, "wb") as stream:
            pickle.dump(self, stream)
        print(f"Model saved at {path}")
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected model type {type(model).__name__}")
        return model


def _new_model(seed):
    return lgb.LGBMRegressor(
        objective="mae",
        learning_rate=0.03,
        num_leaves=63,
        n_estimators=700,
        min_child_samples=30,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=seed,
        verbosity=-1,
        force_col_wise=True,
    )


def _recency_weights(index, half_life_days=180.0):
    age_days = np.asarray(
        (index.max() - index).total_seconds() / 86400.0,
        dtype=float,
    )
    return np.asarray(
        np.power(0.5, age_days / half_life_days),
        dtype=float,
    )


def _fit_window_models(frame, feature_cols, forecast_start, windows):
    history = frame[frame.index < forecast_start]
    valid = clean_mask(history["smp_system_price"].to_numpy())
    history = history[valid]
    models = {}
    for position, window_days in enumerate(windows):
        cutoff = forecast_start - pd.Timedelta(days=window_days)
        window = history[history.index >= cutoff]
        if len(window) < 256:
            continue
        model = _new_model(42 + position)
        model.fit(
            window[feature_cols],
            window["smp_system_price"],
            sample_weight=_recency_weights(window.index),
        )
        models[f"lgb_{window_days}d"] = model
    return models


def _candidate_name(base, bias_days, cap_days, cap_ratio):
    ratio = "off" if cap_ratio is None else f"{cap_ratio:.2f}"
    return f"{base}|b{bias_days}|c{cap_days}|r{ratio}"


def _parse_candidate(name):
    base, bias, cap, ratio = name.split("|")
    return {
        "base_candidate": base,
        "bias_days": int(bias[1:]),
        "cap_days": int(cap[1:]),
        "cap_ratio": None if ratio[1:] == "off" else float(ratio[1:]),
    }


def _pretest_base_predictions(history, year_frame, feature_cols, windows):
    forecast_start = year_frame.index.min().normalize()
    models = _fit_window_models(history, feature_cols, forecast_start, windows)
    if not models:
        raise ValueError(f"No calibration model available for {forecast_start.year}")
    predictions = {
        name: model.predict(year_frame[feature_cols])
        for name, model in models.items()
    }
    prediction_matrix = np.column_stack(list(predictions.values()))
    predictions["adaptive_median"] = np.median(prediction_matrix, axis=1)
    actual = year_frame["smp_system_price"].to_numpy(dtype=float)
    predictions["online_average"] = online_component_average(
        actual,
        year_frame.index,
        {name: values for name, values in predictions.items() if name.startswith("lgb_")},
        lookback_days=ONLINE_WEIGHT_DAYS,
    )

    try:
        multitask = CINGLearForecaster()
        multitask.fit(history[feature_cols], history["smp_system_price"])
        cing_prediction = multitask.predict(year_frame[feature_cols])
        predictions["cing_lear"] = cing_prediction
        predictions["hybrid_median"] = np.median(
            np.column_stack([predictions["adaptive_median"], cing_prediction]),
            axis=1,
        )
    except ValueError as error:
        print(f"CING-LEAR skipped for {forecast_start.year} - {error}")
    return predictions


def run_pretest_validation(
    df,
    feature_cols,
    output_dir=None,
    years=PRETEST_YEARS,
    windows=CALIBRATION_WINDOWS,
):
    years = tuple(int(year) for year in years)
    if not years or max(years) >= 2026:
        raise ValueError("Pretest years must end before 2026")

    required = ["smp_system_price", *feature_cols]
    frame = df.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_index()
    frame = frame[(frame.index.year >= 2021) & (frame.index.year < 2026)]
    rows = []

    for year in years:
        year_frame = frame[frame.index.year == year]
        if year_frame.empty:
            raise ValueError(f"No pretest samples for {year}")
        history = frame[frame.index < pd.Timestamp(year=year, month=1, day=1)]
        base_predictions = _pretest_base_predictions(
            history,
            year_frame,
            feature_cols,
            windows,
        )
        actual = year_frame["smp_system_price"].to_numpy(dtype=float)

        for base_name, base_prediction in base_predictions.items():
            for bias_days, cap_days, cap_ratio in POSTPROCESS_CONFIGS:
                prediction = apply_online_adjustment(
                    base_prediction,
                    year_frame.index,
                    actual,
                    history_actual=history["smp_system_price"],
                    bias_days=bias_days,
                    cap_days=cap_days,
                    cap_ratio=cap_ratio,
                )
                metrics = target_metrics(
                    actual,
                    np.clip(prediction, PRICE_LOWER, PRICE_UPPER),
                )
                rows.append({
                    "year": year,
                    "candidate": _candidate_name(
                        base_name,
                        bias_days,
                        cap_days,
                        cap_ratio,
                    ),
                    **metrics,
                })
        year_rows = pd.DataFrame([row for row in rows if row["year"] == year])
        best = year_rows.loc[
            np.maximum(
                year_rows["clean_mae"] / 150.0,
                year_rows["clean_wmape"] / 10.0,
            ).idxmin()
        ]
        print(
            f"Pretest {year} best {best['candidate']} - "
            f"clean MAE {best['clean_mae']:.2f} "
            f"WMAPE {best['clean_wmape']:.2f}%"
        )

    metrics_frame = pd.DataFrame(rows)
    candidate_scores = {}
    for name, values in metrics_frame.groupby("candidate"):
        if len(values) != len(years):
            continue
        target_ratio = np.maximum(
            values["clean_mae"].to_numpy() / 150.0,
            values["clean_wmape"].to_numpy() / 10.0,
        )
        candidate_scores[name] = float(target_ratio.max())

    selected = min(candidate_scores, key=candidate_scores.get)
    selected_rows = metrics_frame[metrics_frame["candidate"] == selected]
    target_met = bool(
        (selected_rows["clean_mae"] < 150.0).all()
        and (selected_rows["clean_wmape"] < 10.0).all()
    )
    summary = {
        "years": list(years),
        "windows": list(windows),
        "selected_candidate": selected,
        **_parse_candidate(selected),
        "worst_target_ratio": candidate_scores[selected],
        "target_met": target_met,
    }
    print(
        f"Pretest selected {selected} - target met {target_met} "
        f"worst ratio {candidate_scores[selected]:.3f}"
    )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_frame.to_csv(output_path / "pretest_metrics.csv", index=False)
        with open(output_path / "pretest_summary.json", "w") as stream:
            json.dump(summary, stream, indent=2)

    return summary, metrics_frame


def train_adaptive_model(
    df,
    feature_cols,
    validation_summary,
    output_dir="outputs/models",
    model_name="adaptive_window.pkl",
):
    required = ["smp_system_price", *feature_cols]
    frame = df.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_index()
    frame = frame[(frame.index.year >= 2021) & (frame.index.year <= 2025)]
    X = frame[feature_cols]
    y = frame["smp_system_price"]
    model = AdaptiveWindowEnsemble(
        selected_candidate=validation_summary["base_candidate"],
        windows=validation_summary["windows"],
        bias_days=validation_summary["bias_days"],
        cap_days=validation_summary["cap_days"],
        cap_ratio=validation_summary["cap_ratio"],
    )
    model.selected_features = list(feature_cols)
    model.fit(X, y)
    model.save(output_dir, model_name=model_name)
    return model, list(feature_cols)
