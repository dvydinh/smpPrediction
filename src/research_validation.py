import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


PRETEST_YEARS = (2024, 2025)
CALIBRATION_WINDOWS = (365, 730, 1460)


class AdaptiveWindowEnsemble:
    def __init__(
        self,
        selected_candidate="adaptive_median",
        windows=CALIBRATION_WINDOWS,
    ):
        self.selected_candidate_ = selected_candidate
        self.windows = tuple(int(value) for value in windows)
        self.models = {}
        self.model_order_ = []
        self.selected_features = []

    def fit(self, X, y):
        frame = X.copy()
        frame["smp_system_price"] = np.asarray(y, dtype=float)
        frame = frame.sort_index()
        clean = (
            (frame["smp_system_price"] > 500.0)
            & (frame["smp_system_price"] < 2500.0)
        )
        frame = frame[clean]
        forecast_start = X.index.max() + pd.Timedelta(minutes=30)
        for position, window_days in enumerate(self.windows):
            cutoff = forecast_start - pd.Timedelta(days=window_days)
            window = frame[frame.index >= cutoff]
            model = _new_model(42 + position)
            model.fit(
                window[self.selected_features],
                window["smp_system_price"],
                sample_weight=_recency_weights(window.index),
            )
            name = f"lgb_{window_days}d"
            self.models[name] = model
            self.model_order_.append(name)
        self.models["lgb"] = self.models[self.model_order_[-1]]
        return self

    def predict(self, X):
        predictions = {
            name: self.models[name].predict(X)
            for name in self.model_order_
        }
        if self.selected_candidate_ == "adaptive_median":
            return np.median(
                np.column_stack(list(predictions.values())),
                axis=1,
            )
        if self.selected_candidate_ not in predictions:
            raise ValueError(
                f"Unknown adaptive candidate {self.selected_candidate_}"
            )
        return predictions[self.selected_candidate_]

    def save(self, output_dir, model_name="adaptive_window.pkl"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, model_name)
        with open(path, "wb") as stream:
            pickle.dump(self, stream)
        print(f"Model saved at {path}")
        return path


def _new_model(seed):
    return lgb.LGBMRegressor(
        objective="mae",
        learning_rate=0.02,
        num_leaves=63,
        n_estimators=900,
        min_child_samples=40,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=seed,
        verbosity=-1,
        force_col_wise=True,
    )


def _recency_weights(index, half_life_days=365.0):
    age_days = (index.max() - index).total_seconds() / 86400.0
    return np.power(0.5, age_days / half_life_days)


def _metrics(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = np.abs(actual - prediction)
    denominator = np.abs(actual).sum()
    clean = (actual > 500.0) & (actual < 2500.0)
    clean_error = error[clean]
    clean_denominator = np.abs(actual[clean]).sum()
    return {
        "samples": int(len(actual)),
        "mae": float(error.mean()),
        "wmape": float(100.0 * error.sum() / denominator),
        "clean_samples": int(clean.sum()),
        "clean_mae": float(clean_error.mean()),
        "clean_wmape": float(
            100.0 * clean_error.sum() / clean_denominator
        ),
    }


def _fit_window_models(frame, feature_cols, forecast_start, windows):
    history = frame[frame.index < forecast_start]
    clean = (
        (history["smp_system_price"] > 500.0)
        & (history["smp_system_price"] < 2500.0)
    )
    history = history[clean]
    models = {}
    for position, window_days in enumerate(windows):
        cutoff = forecast_start - pd.Timedelta(days=window_days)
        window = history[history.index >= cutoff]
        if len(window) < 48 * 180:
            raise ValueError(
                f"Calibration window {window_days} has only {len(window)} rows"
            )
        model = _new_model(42 + position)
        model.fit(
            window[feature_cols],
            window["smp_system_price"],
            sample_weight=_recency_weights(window.index),
        )
        models[f"lgb_{window_days}d"] = model
    return models


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
        forecast_start = pd.Timestamp(year=year, month=1, day=1)
        window_models = _fit_window_models(
            frame,
            feature_cols,
            forecast_start,
            windows,
        )
        predictions = {
            name: model.predict(year_frame[feature_cols])
            for name, model in window_models.items()
        }

        prediction_matrix = np.column_stack(list(predictions.values()))
        predictions["adaptive_median"] = np.median(prediction_matrix, axis=1)
        if "smp_same_cycle_1d" in year_frame.columns:
            predictions["seasonal_baseline"] = year_frame[
                "smp_same_cycle_1d"
            ].to_numpy(dtype=float)

        actual = year_frame["smp_system_price"].to_numpy(dtype=float)
        for name, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite pretest predictions for {name} {year}")
            metrics = _metrics(actual, np.clip(prediction, 0.0, 1778.6))
            rows.append({"year": year, "candidate": name, **metrics})
            print(
                f"Pretest {year} {name} - clean MAE {metrics['clean_mae']:.2f} "
                f"WMAPE {metrics['clean_wmape']:.2f}%"
            )

    metrics_frame = pd.DataFrame(rows)
    candidates = [
        name
        for name in metrics_frame["candidate"].unique()
        if name != "seasonal_baseline"
    ]
    candidate_scores = {}
    for name in candidates:
        values = metrics_frame[metrics_frame["candidate"] == name]
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
    selected_candidate,
    output_dir="outputs/models",
    model_name="adaptive_window.pkl",
):
    required = ["smp_system_price", *feature_cols]
    frame = df.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_index()
    frame = frame[(frame.index.year >= 2021) & (frame.index.year <= 2025)]
    X = frame[feature_cols]
    y = frame["smp_system_price"]
    model = AdaptiveWindowEnsemble(selected_candidate=selected_candidate)
    model.selected_features = list(feature_cols)
    model.fit(X, y)
    model.save(output_dir, model_name=model_name)
    return model, list(feature_cols)
