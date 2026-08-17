import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import RobustScaler


PROFILE_COLUMNS = (
    "smp_same_cycle_1d",
    "smp_same_cycle_2d",
    "smp_same_cycle_3d",
    "smp_same_cycle_7d",
    "smp_same_cycle_14d",
    "smp_same_cycle_28d",
    "load_same_cycle_1d",
    "load_same_cycle_2d",
    "load_same_cycle_3d",
    "load_same_cycle_7d",
    "load_same_cycle_14d",
    "residual_load_proxy",
    "load_forecast_proxy",
    "load_forecast_trend",
    "thermal_margin_proxy",
    "solar_gen_proxy",
    "wind_gen_proxy",
)

PROFILE_PREFIXES = (
    "temperature_",
    "humidity_",
    "cloud_cover_",
    "wind_speed_",
    "shortwave_radiation_",
)

SCALAR_COLUMNS = (
    "is_weekend",
    "is_workday",
    "is_holiday",
    "is_tet",
    "is_pre_holiday",
    "is_post_holiday",
    "is_post_covid",
    "season",
    "morning_smp_mean",
    "morning_smp_max",
    "morning_smp_min",
    "morning_load_mean",
    "prev_full_smp_mean",
    "prev_full_smp_max",
    "prev_full_smp_min",
    "prev_full_gate_prob",
    "smp_rolling_std_1d",
    "smp_rolling_std_7d",
    "smp_rolling_mean_7d",
)


def _cycle_index(index):
    return index.hour * 2 + (index.minute == 30).astype(int)


class DailyMatrixBuilder:
    def __init__(self):
        self.profile_columns = None
        self.scalar_columns = None
        self.output_columns = None

    def fit(self, X):
        self.profile_columns = [
            col for col in X.columns
            if col in PROFILE_COLUMNS or col.startswith(PROFILE_PREFIXES)
        ]
        self.scalar_columns = [col for col in SCALAR_COLUMNS if col in X.columns]
        matrix = self._transform(X)
        self.output_columns = list(matrix.columns)
        return self

    def transform(self, X):
        matrix = self._transform(X)
        return matrix.reindex(columns=self.output_columns, fill_value=0.0)

    def _transform(self, X):
        frame = X.copy()
        frame["_forecast_date"] = frame.index.normalize()
        frame["_cycle"] = _cycle_index(frame.index)
        rows = []
        dates = []

        for forecast_date, group in frame.groupby("_forecast_date", sort=True):
            group = group.sort_values("_cycle").drop_duplicates("_cycle", keep="last")
            group = group.set_index("_cycle").reindex(range(48))
            values = {}
            for col in self.profile_columns:
                series = pd.to_numeric(group[col], errors="coerce").ffill().bfill().fillna(0.0)
                for cycle, value in enumerate(series.to_numpy(dtype=float)):
                    values[f"{col}_c{cycle:02d}"] = value
            for col in self.scalar_columns:
                series = pd.to_numeric(group[col], errors="coerce").dropna()
                values[col] = float(series.iloc[0]) if len(series) else 0.0
            rows.append(values)
            dates.append(forecast_date)

        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), dtype=float).fillna(0.0)


def _daily_targets(y):
    frame = pd.DataFrame({"target": y.astype(float)})
    frame["forecast_date"] = frame.index.normalize()
    frame["cycle"] = _cycle_index(frame.index)
    daily = frame.pivot_table(
        index="forecast_date", columns="cycle", values="target", aggfunc="last"
    ).reindex(columns=range(48))
    return daily.dropna(axis=0, how="any")


def _rows_from_daily(predictions, row_index):
    output = np.empty(len(row_index), dtype=float)
    dates = row_index.normalize()
    cycles = _cycle_index(row_index)
    for i, (date, cycle) in enumerate(zip(dates, cycles)):
        output[i] = predictions.loc[date, cycle]
    return output


class CINGLearForecaster:
    """Multi-output group-sparse day-ahead model inspired by CING-LEAR."""

    def __init__(self, windows=(365, 730, 1095, None), alphas=(0.0003, 0.001, 0.003)):
        self.windows = windows
        self.alphas = alphas
        self.builder = DailyMatrixBuilder()
        self.members = []
        self.weights = None

    def fit(self, X, y):
        daily_X = self.builder.fit(X).transform(X)
        daily_y = _daily_targets(y)
        common = daily_X.index.intersection(daily_y.index).sort_values()
        daily_X = daily_X.loc[common]
        daily_y = daily_y.loc[common]
        self.members = []
        validation_errors = []
        fitted_lengths = set()

        for window in self.windows:
            window_X = daily_X if window is None else daily_X.tail(window)
            window_y = daily_y.loc[window_X.index]
            if len(window_X) < 120 or len(window_X) in fitted_lengths:
                continue
            fitted_lengths.add(len(window_X))

            validation_days = min(84, max(28, len(window_X) // 6))
            fit_X = window_X.iloc[:-validation_days]
            fit_y = window_y.iloc[:-validation_days]
            val_X = window_X.iloc[-validation_days:]
            val_y = window_y.iloc[-validation_days:]

            best_alpha = self.alphas[0]
            best_error = np.inf
            for alpha in self.alphas:
                candidate = self._fit_member(fit_X, fit_y, alpha)
                pred = self._predict_member(candidate, val_X)
                error = np.mean(np.abs(val_y.to_numpy() - pred))
                if error < best_error:
                    best_alpha = alpha
                    best_error = error

            member = self._fit_member(window_X, window_y, best_alpha)
            self.members.append(member)
            validation_errors.append(max(best_error, 1e-6))

        if not self.members:
            raise ValueError("CING-LEAR requires at least 120 complete training days")

        inverse = 1.0 / np.asarray(validation_errors)
        self.weights = inverse / inverse.sum()
        return self

    def _fit_member(self, X, y, alpha):
        x_scaler = RobustScaler(quantile_range=(10.0, 90.0))
        X_scaled = x_scaler.fit_transform(X)
        y_center = y.median(axis=0).to_numpy(dtype=float)
        y_scale = (y.quantile(0.9) - y.quantile(0.1)).to_numpy(dtype=float)
        y_scale = np.where(y_scale > 1e-6, y_scale, 1.0)
        y_scaled = (y.to_numpy(dtype=float) - y_center) / y_scale
        model = MultiTaskLasso(
            alpha=alpha,
            max_iter=8000,
            tol=1e-5,
            selection="random",
            random_state=42,
        )
        model.fit(X_scaled, y_scaled)
        return x_scaler, y_center, y_scale, model

    @staticmethod
    def _predict_member(member, X):
        x_scaler, y_center, y_scale, model = member
        return model.predict(x_scaler.transform(X)) * y_scale + y_center

    def predict(self, X):
        daily_X = self.builder.transform(X)
        member_predictions = [self._predict_member(member, daily_X) for member in self.members]
        combined = np.average(np.stack(member_predictions), axis=0, weights=self.weights)
        daily = pd.DataFrame(combined, index=daily_X.index, columns=range(48))
        return _rows_from_daily(daily, X.index)


class SimilarDayForecaster:
    """Distance-weighted daily profile expert using only forecast-safe inputs."""

    def __init__(self, neighbors=21):
        self.neighbors = neighbors
        self.builder = DailyMatrixBuilder()
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        daily_X = self.builder.fit(X).transform(X)
        daily_y = _daily_targets(y)
        common = daily_X.index.intersection(daily_y.index).sort_values()
        daily_X = daily_X.loc[common]
        daily_y = daily_y.loc[common]
        self.train_X = self.scaler.fit_transform(daily_X)
        self.train_y = daily_y.to_numpy(dtype=float)
        return self

    def predict(self, X):
        daily_X = self.builder.transform(X)
        query = self.scaler.transform(daily_X)
        predictions = []
        k = min(self.neighbors, len(self.train_X))
        for row in query:
            distances = np.mean((self.train_X - row) ** 2, axis=1)
            nearest = np.argpartition(distances, k - 1)[:k]
            local = distances[nearest]
            scale = np.median(local) + 1e-9
            weights = np.exp(-local / scale)
            weights = weights / weights.sum()
            predictions.append(np.average(self.train_y[nearest], axis=0, weights=weights))
        daily = pd.DataFrame(predictions, index=daily_X.index, columns=range(48))
        return _rows_from_daily(daily, X.index)
