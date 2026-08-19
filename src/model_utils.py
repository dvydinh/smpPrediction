import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from src.daily_models import CINGLearForecaster, SimilarDayForecaster
from src.regime_models import RegimeForecaster


def _cycle_values(index):
    return np.asarray(
        index.hour * 2 + (index.minute == 30).astype(int),
        dtype=int,
    )


class StackingEnsemble:
    def __init__(self, output_dir, recency_half_life_days=730):
        self.output_dir = output_dir
        self.recency_half_life_days = recency_half_life_days
        self.models = {}
        self.model_order_ = []
        self.meta_learner = None
        self.meta_kind_ = "global"
        self.selected_candidate_ = None
        self.cycle_bias_ = np.zeros(48, dtype=float)
        self.base_selection_scores_ = {}
        self.regime_estimators_ = {
            "classifier": 800,
            "normal": 800,
            "low": 800,
        }
        self.gate_threshold_ = 1.1
        self.gate_ramp_ = 0.05
        self.gate_validation_scores_ = {}
        self.gate_classifier_ = None
        self.normal_delta_bounds_ = np.tile([-550.0, 550.0], (48, 1))
        self.gate_delta_bounds_ = np.tile([-1550.0, 1550.0], (48, 1))
        self.shape_guard_enabled_ = True
        self.state_projection_enabled_ = False
        self.lower_projection_cut_ = None
        self.cap_projection_cut_ = None
        self.lower_state_value_ = 1000.0
        self.cap_state_value_ = 1725.2
        self.cap_projection_fallback_ = False
        self.state_projection_scores_ = {}

    def get_base_models(self):
        return {
            "lgb": lgb.LGBMRegressor(
                objective="mae",
                learning_rate=0.005,
                num_leaves=255,
                n_estimators=3000,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
            ),
            "xgb": xgb.XGBRegressor(
                objective="reg:absoluteerror",
                learning_rate=0.005,
                max_depth=10,
                n_estimators=3000,
                colsample_bytree=0.7,
                subsample=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=42,
                tree_method="hist",
                device="cuda",
            ),
            "cb": CatBoostRegressor(
                loss_function="MAE",
                learning_rate=0.01,
                iterations=3000,
                depth=10,
                l2_leaf_reg=5,
                random_seed=42,
                verbose=False,
                task_type="GPU",
            ),
        }

    def _recency_weights(self, index):
        age_days = (index.max() - index).total_seconds() / 86400.0
        weights = np.power(0.5, age_days / self.recency_half_life_days)
        return np.asarray(weights, dtype=float)

    def _fit_tree(self, name, model, X_train, y_train, X_val=None, y_val=None):
        train_weight = self._recency_weights(X_train.index)
        if X_val is None:
            model.fit(X_train, y_train, sample_weight=train_weight)
            return model

        val_weight = self._recency_weights(X_val.index)
        if name == "lgb":
            model.fit(
                X_train,
                y_train,
                sample_weight=train_weight,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[val_weight],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif name == "xgb":
            model.fit(
                X_train,
                y_train,
                sample_weight=train_weight,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[val_weight],
                verbose=False,
            )
        else:
            model.fit(
                X_train,
                y_train,
                sample_weight=train_weight,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False,
            )
        return model

    def fit(self, X, y):
        X = X.sort_index()
        y = y.loc[X.index]
        days = pd.DatetimeIndex(X.index.normalize().unique()).sort_values()
        day_splitter = TimeSeriesSplit(n_splits=5)
        model_names = [
            "lgb",
            "xgb",
            "cb",
            "cing_lear",
            "similar_day",
            "regime",
        ]
        oof_predictions = np.full((len(X), len(model_names)), np.nan, dtype=float)
        oof_low_probability = np.full(len(X), np.nan, dtype=float)
        oof_low_prediction = np.full(len(X), np.nan, dtype=float)
        regime_iteration_history = {name: [] for name in self.regime_estimators_}

        print("Stage 1 - chronological out-of-fold base forecasts")
        for fold, (train_day_idx, val_day_idx) in enumerate(day_splitter.split(days), 1):
            train_days = days[train_day_idx]
            val_days = days[val_day_idx]
            train_positions = np.flatnonzero(X.index.normalize().isin(train_days))
            val_positions = np.flatnonzero(X.index.normalize().isin(val_days))
            X_train, y_train = X.iloc[train_positions], y.iloc[train_positions]
            X_val, y_val = X.iloc[val_positions], y.iloc[val_positions]

            print(f"  fold {fold}/5")
            for column, name in enumerate(("lgb", "xgb", "cb")):
                model = self.get_base_models()[name]
                model = self._fit_tree(name, model, X_train, y_train, X_val, y_val)
                oof_predictions[val_positions, column] = model.predict(X_val)

            cing_lear = CINGLearForecaster()
            cing_lear.fit(X_train, y_train)
            oof_predictions[val_positions, 3] = cing_lear.predict(X_val)

            similar_day = SimilarDayForecaster(neighbors=21)
            similar_day.fit(X_train, y_train)
            oof_predictions[val_positions, 4] = similar_day.predict(X_val)

            print("    two-regime expert")
            regime = RegimeForecaster()
            regime.fit(
                X_train,
                y_train,
                sample_weight=self._recency_weights(X_train.index),
                X_val=X_val,
                y_val=y_val,
                val_weight=self._recency_weights(X_val.index),
            )
            probability, normal_prediction, low_prediction = regime.predict_components(X_val)
            oof_predictions[val_positions, 5] = (
                (1.0 - probability) * normal_prediction
                + probability * low_prediction
            )
            oof_low_probability[val_positions] = probability
            oof_low_prediction[val_positions] = low_prediction
            for name, iteration in regime.best_iterations_.items():
                regime_iteration_history[name].append(iteration)

        self.gate_classifier_ = regime.classifier

        for name, history in regime_iteration_history.items():
            if history:
                self.regime_estimators_[name] = int(np.clip(
                    np.median(history),
                    100,
                    800,
                ))
        print(f"  regime estimators {self.regime_estimators_}")

        valid_positions = np.flatnonzero(
            np.isfinite(oof_predictions).all(axis=1)
            & np.isfinite(oof_low_probability)
            & np.isfinite(oof_low_prediction)
        )
        meta_X = oof_predictions[valid_positions]
        meta_y = y.iloc[valid_positions]
        candidate_sets = {
            "trees": [0, 1, 2],
            "trees_cing": [0, 1, 2, 3],
            "trees_similar": [0, 1, 2, 4],
        }
        meta_values = meta_y.to_numpy(dtype=float)
        meta_cycles = _cycle_values(X.index[valid_positions])
        clean_meta = (meta_values > 500.0) & (meta_values < 2500.0)
        selection_cut = int(len(meta_y) * 0.8)
        train_selection = clean_meta & (np.arange(len(meta_y)) < selection_cut)
        validation_selection = np.arange(len(meta_y)) >= selection_cut
        clean_validation = clean_meta[validation_selection]
        best_name = None
        best_score = np.inf
        best_validation_pred = None
        for candidate_name, columns in candidate_sets.items():
            candidate = self._new_meta_learner()
            candidate.fit(
                meta_X[train_selection][:, columns],
                meta_values[train_selection],
            )
            validation_pred = candidate.predict(
                meta_X[validation_selection][:, columns]
            )
            validation_y = meta_values[validation_selection]
            score = np.mean(np.abs(
                validation_y[clean_validation]
                - validation_pred[clean_validation]
            ))
            self.base_selection_scores_[candidate_name] = float(score)
            print(f"  candidate {candidate_name} clean mae {score:.3f}")
            if score < best_score:
                best_name = candidate_name
                best_score = score
                best_validation_pred = validation_pred

        selected_columns = candidate_sets[best_name]
        self.selected_candidate_ = best_name
        self.model_order_ = [model_names[column] for column in selected_columns]
        self.meta_learner = self._new_meta_learner()
        self.meta_learner.fit(
            meta_X[clean_meta][:, selected_columns],
            meta_values[clean_meta],
        )

        validation_cycles = meta_cycles[validation_selection]
        validation_values = meta_values[validation_selection]
        validation_residuals = validation_values - best_validation_pred
        for cycle in range(48):
            local_mask = (validation_cycles == cycle) & clean_validation
            local = validation_residuals[local_mask]
            if len(local):
                shrinkage = len(local) / (len(local) + 96.0)
                self.cycle_bias_[cycle] = np.clip(
                    shrinkage * np.median(local),
                    -40.0,
                    40.0,
                )

        normal_validation = best_validation_pred + self.cycle_bias_[validation_cycles]
        validation_probability = oof_low_probability[valid_positions][validation_selection]
        validation_low = np.clip(
            oof_low_prediction[valid_positions][validation_selection],
            0.0,
            500.0,
        )
        self._tune_gate(
            validation_values,
            normal_validation,
            validation_probability,
            validation_low,
        )
        self._fit_delta_bounds(y)
        gate_weight = self._gate_weight(validation_probability)
        gated_validation = (
            (1.0 - gate_weight) * normal_validation
            + gate_weight * validation_low
        )
        guarded_validation = self._apply_shape_guard(
            gated_validation,
            gate_weight,
            X.index[valid_positions][validation_selection],
        )
        raw_clean_mae = np.mean(np.abs(
            validation_values[clean_validation]
            - gated_validation[clean_validation]
        ))
        guarded_clean_mae = np.mean(np.abs(
            validation_values[clean_validation]
            - guarded_validation[clean_validation]
        ))
        raw_mae = np.mean(np.abs(validation_values - gated_validation))
        guarded_mae = np.mean(np.abs(validation_values - guarded_validation))
        self.shape_guard_enabled_ = bool(
            guarded_clean_mae <= raw_clean_mae + 1.0
            and guarded_mae <= raw_mae + 1.0
        )
        projection_base = (
            guarded_validation
            if self.shape_guard_enabled_
            else gated_validation
        )
        self._tune_state_projection(
            validation_values,
            projection_base,
            gate_weight,
            X.index[valid_positions][validation_selection],
            y,
        )

        print("Stage 2 - full-data base models")
        for name in ("lgb", "xgb", "cb"):
            model = self.get_base_models()[name]
            self.models[name] = self._fit_tree(name, model, X, y)

        if "cing_lear" in self.model_order_:
            self.models["cing_lear"] = CINGLearForecaster().fit(X, y)
        if "similar_day" in self.model_order_:
            self.models["similar_day"] = SimilarDayForecaster(neighbors=21).fit(X, y)
        self.models["regime"] = RegimeForecaster(
            n_estimators=self.regime_estimators_,
        ).fit(X, y, sample_weight=self._recency_weights(X.index))
        print(f"Selected base models - {self.model_order_}")
        print(f"Selected meta model - {self.meta_kind_}")
        print(
            f"Selected collapse gate - threshold {self.gate_threshold_:.2f} "
            f"ramp {self.gate_ramp_:.2f}"
        )
        print(f"Shape guard enabled - {self.shape_guard_enabled_}")
        print(
            f"State projection - enabled {self.state_projection_enabled_} "
            f"lower {self.lower_projection_cut_} cap {self.cap_projection_cut_}"
        )
        print("Stacking ensemble training complete")
        return self

    @staticmethod
    def _new_meta_learner():
        return RidgeCV(
            alphas=np.logspace(-2, 5, 20),
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=5),
        )

    @staticmethod
    def _gate_weight_for(probability, threshold, ramp):
        probability = np.asarray(probability, dtype=float)
        if threshold > 1.0:
            return np.zeros(len(probability), dtype=float)
        if ramp <= 0.0:
            return (probability >= threshold).astype(float)
        lower = threshold - ramp
        upper = threshold + ramp
        return np.clip((probability - lower) / (upper - lower), 0.0, 1.0)

    def _gate_weight(self, probability):
        return self._gate_weight_for(
            probability,
            self.gate_threshold_,
            self.gate_ramp_,
        )

    def predict_gate_probability(self, X):
        if self.gate_classifier_ is not None:
            return self.gate_classifier_.predict_proba(X)[:, 1]
        probability, _, _ = self.models["regime"].predict_components(X)
        return probability

    def _tune_gate(self, actual, normal_prediction, probability, low_prediction):
        actual = np.asarray(actual, dtype=float)
        normal_prediction = np.asarray(normal_prediction, dtype=float)
        probability = np.asarray(probability, dtype=float)
        low_prediction = np.asarray(low_prediction, dtype=float)
        clean = (actual > 500.0) & (actual < 2500.0)
        low = actual <= 500.0
        base_clean_mae = np.mean(np.abs(actual[clean] - normal_prediction[clean]))
        best_overall_mae = np.mean(np.abs(actual - normal_prediction))
        self.gate_validation_scores_ = {
            "off": {
                "clean_mae": float(base_clean_mae),
                "overall_mae": float(best_overall_mae),
                "low_mae": float(np.mean(np.abs(
                    actual[low] - normal_prediction[low]
                ))),
            }
        }
        self.gate_threshold_ = 1.1
        self.gate_ramp_ = 0.05

        for threshold in np.arange(0.20, 0.96, 0.05):
            for ramp in (0.0, 0.05, 0.10):
                weight = self._gate_weight_for(probability, threshold, ramp)
                prediction = (
                    (1.0 - weight) * normal_prediction
                    + weight * low_prediction
                )
                clean_mae = np.mean(np.abs(actual[clean] - prediction[clean]))
                overall_mae = np.mean(np.abs(actual - prediction))
                low_mae = np.mean(np.abs(actual[low] - prediction[low]))
                name = f"t{threshold:.2f}_r{ramp:.2f}"
                self.gate_validation_scores_[name] = {
                    "clean_mae": float(clean_mae),
                    "overall_mae": float(overall_mae),
                    "low_mae": float(low_mae),
                }
                if (
                    clean_mae <= base_clean_mae + 2.5
                    and overall_mae < best_overall_mae
                ):
                    best_overall_mae = overall_mae
                    self.gate_threshold_ = float(threshold)
                    self.gate_ramp_ = float(ramp)

    @staticmethod
    def _project_states(
        prediction,
        gate_weight,
        lower_cut,
        cap_cut,
        lower_value,
        cap_value,
    ):
        result = np.asarray(prediction, dtype=float).copy()
        normal = np.asarray(gate_weight, dtype=float) < 0.20
        if lower_cut is not None:
            result[normal & (result <= lower_cut)] = lower_value
        if cap_cut is not None:
            result[normal & (result >= cap_cut)] = cap_value
        return result

    def _tune_state_projection(
        self,
        actual,
        prediction,
        gate_weight,
        index,
        training_target,
    ):
        actual = np.asarray(actual, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        gate_weight = np.asarray(gate_weight, dtype=float)
        training_values = np.asarray(training_target, dtype=float)
        lower_values = training_values[
            (training_values > 500.0) & (training_values < 1200.0)
        ]
        cap_values = training_values[training_values >= 1700.0]
        if len(lower_values) < 100 or len(cap_values) < 100:
            return

        self.lower_state_value_ = float(np.median(lower_values))
        self.cap_state_value_ = float(np.median(cap_values))
        clean = (actual > 500.0) & (actual < 2500.0)
        base_clean_mae = float(np.mean(np.abs(
            actual[clean] - prediction[clean]
        )))
        base_overall_mae = float(np.mean(np.abs(actual - prediction)))
        months = pd.PeriodIndex(pd.DatetimeIndex(index), freq="M")
        unique_months = months.unique()
        base_month_mae = {
            month: float(np.mean(np.abs(
                actual[clean & (months == month)]
                - prediction[clean & (months == month)]
            )))
            for month in unique_months
            if np.any(clean & (months == month))
        }
        self.state_projection_scores_ = {
            "off": {
                "clean_mae": base_clean_mae,
                "overall_mae": base_overall_mae,
            }
        }
        best = None
        lower_candidates = [None, *np.arange(1150.0, 1451.0, 25.0)]
        cap_candidates = [None, *np.arange(1450.0, 1701.0, 25.0)]
        required_months = max(1, int(np.ceil(len(base_month_mae) * 0.60)))

        for lower_cut in lower_candidates:
            for cap_cut in cap_candidates:
                if lower_cut is None and cap_cut is None:
                    continue
                if (
                    lower_cut is not None
                    and cap_cut is not None
                    and lower_cut >= cap_cut
                ):
                    continue
                projected = self._project_states(
                    prediction,
                    gate_weight,
                    lower_cut,
                    cap_cut,
                    self.lower_state_value_,
                    self.cap_state_value_,
                )
                clean_mae = float(np.mean(np.abs(
                    actual[clean] - projected[clean]
                )))
                overall_mae = float(np.mean(np.abs(actual - projected)))
                improved_months = sum(
                    float(np.mean(np.abs(
                        actual[clean & (months == month)]
                        - projected[clean & (months == month)]
                    ))) <= month_mae
                    for month, month_mae in base_month_mae.items()
                )
                if (
                    clean_mae < base_clean_mae
                    and overall_mae <= base_overall_mae
                    and improved_months >= required_months
                    and (best is None or clean_mae < best[0])
                ):
                    best = (
                        clean_mae,
                        overall_mae,
                        lower_cut,
                        cap_cut,
                        improved_months,
                    )

        if best is None:
            clean_mae = base_clean_mae
            overall_mae = base_overall_mae
            lower_cut = None
            cap_cut = None
            improved_months = 0
        else:
            clean_mae, overall_mae, lower_cut, cap_cut, improved_months = best

        cap_tolerance = max(5.0, self.cap_state_value_ * 0.005)
        cap_atom_share = float(np.mean(
            np.abs(cap_values - self.cap_state_value_) <= cap_tolerance
        ))
        if cap_cut is None and cap_atom_share >= 0.50:
            cap_cut = 0.90 * self.cap_state_value_
            self.cap_projection_fallback_ = True
            projected = self._project_states(
                prediction,
                gate_weight,
                lower_cut,
                cap_cut,
                self.lower_state_value_,
                self.cap_state_value_,
            )
            clean_mae = float(np.mean(np.abs(
                actual[clean] - projected[clean]
            )))
            overall_mae = float(np.mean(np.abs(actual - projected)))
            improved_months = sum(
                float(np.mean(np.abs(
                    actual[clean & (months == month)]
                    - projected[clean & (months == month)]
                ))) <= month_mae
                for month, month_mae in base_month_mae.items()
            )

        if lower_cut is None and cap_cut is None:
            return
        self.state_projection_enabled_ = True
        self.lower_projection_cut_ = (
            None if lower_cut is None else float(lower_cut)
        )
        self.cap_projection_cut_ = None if cap_cut is None else float(cap_cut)
        self.state_projection_scores_["selected"] = {
            "clean_mae": float(clean_mae),
            "overall_mae": float(overall_mae),
            "improved_months": int(improved_months),
            "evaluated_months": int(len(base_month_mae)),
            "cap_fallback": bool(self.cap_projection_fallback_),
            "cap_atom_share": float(cap_atom_share),
        }

    def _apply_state_projection(self, prediction, gate_weight):
        if not self.state_projection_enabled_:
            return np.asarray(prediction, dtype=float)
        return self._project_states(
            prediction,
            gate_weight,
            self.lower_projection_cut_,
            self.cap_projection_cut_,
            self.lower_state_value_,
            self.cap_state_value_,
        )

    def _fit_delta_bounds(self, y):
        frame = pd.DataFrame({"target": y.astype(float)})
        frame["previous"] = frame["target"].shift(1)
        frame["delta"] = frame["target"] - frame["previous"]
        frame["cycle"] = _cycle_values(frame.index)
        dates = frame.index.normalize()
        frame["same_day"] = np.r_[False, dates[1:] == dates[:-1]]
        normal = (
            frame["same_day"]
            & (frame["target"] > 500.0)
            & (frame["previous"] > 500.0)
        )
        gate = frame["same_day"] & ~normal
        global_normal = frame.loc[normal, "delta"].quantile([0.01, 0.99]).to_numpy()
        global_gate = frame.loc[gate, "delta"].quantile([0.005, 0.995]).to_numpy()

        for cycle in range(48):
            cycle_mask = frame["cycle"] == cycle
            normal_delta = frame.loc[normal & cycle_mask, "delta"]
            gate_delta = frame.loc[gate & cycle_mask, "delta"]
            normal_bounds = (
                normal_delta.quantile([0.01, 0.99]).to_numpy()
                if len(normal_delta) >= 50 else global_normal
            )
            gate_bounds = (
                gate_delta.quantile([0.005, 0.995]).to_numpy()
                if len(gate_delta) >= 50 else global_gate
            )
            self.normal_delta_bounds_[cycle] = [
                min(float(normal_bounds[0]), -100.0),
                max(float(normal_bounds[1]), 100.0),
            ]
            self.gate_delta_bounds_[cycle] = [
                min(float(gate_bounds[0]), -500.0),
                max(float(gate_bounds[1]), 500.0),
            ]

    def _apply_shape_guard(self, prediction, gate_weight, index):
        result = np.asarray(prediction, dtype=float).copy()
        gate_weight = np.asarray(gate_weight, dtype=float)
        index = pd.DatetimeIndex(index)
        cycles = _cycle_values(index)
        dates = index.normalize()
        for position in range(1, len(result)):
            if dates[position] != dates[position - 1]:
                continue
            gate_transition = (
                max(gate_weight[position], gate_weight[position - 1]) >= 0.20
                or abs(gate_weight[position] - gate_weight[position - 1]) >= 0.20
            )
            bounds = (
                self.gate_delta_bounds_[cycles[position]]
                if gate_transition else self.normal_delta_bounds_[cycles[position]]
            )
            delta = result[position] - result[position - 1]
            result[position] = result[position - 1] + np.clip(
                delta,
                bounds[0],
                bounds[1],
            )
        return result

    def predict(self, X):
        predictions = [self.models[name].predict(X) for name in self.model_order_]
        base_predictions = np.column_stack(predictions)
        cycles = _cycle_values(X.index)
        normal_prediction = (
            self.meta_learner.predict(base_predictions)
            + self.cycle_bias_[cycles]
        )
        probability = self.predict_gate_probability(X)
        _, _, low_prediction = self.models["regime"].predict_components(X)
        gate_weight = self._gate_weight(probability)
        result = (
            (1.0 - gate_weight) * normal_prediction
            + gate_weight * np.clip(low_prediction, 0.0, 500.0)
        )
        if self.shape_guard_enabled_:
            result = self._apply_shape_guard(result, gate_weight, X.index)
        return self._apply_state_projection(result, gate_weight)

    def save(self, model_name="stacking_ensemble.pkl"):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, model_name)
        with open(path, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved at {path}")
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected model type {type(model).__name__}")
        return model


def prepare_training_data(df, feature_cols):
    clean = df.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["smp_system_price"] + feature_cols).copy()
    clean = clean[clean.index.year >= 2021].sort_index()
    X = clean[feature_cols]
    y = clean["smp_system_price"]
    print(f"Data shape - X {X.shape} - y {y.shape}")
    return X, y


def train_and_save_model(
    df,
    feature_cols,
    output_dir="outputs/models",
    model_name="stacking_ensemble.pkl",
):
    X, y = prepare_training_data(df, feature_cols)
    train_mask = X.index.year <= 2025
    X_train, y_train = X[train_mask], y[train_mask]
    print(f"Training data - {len(X_train)} rows - {X_train.shape[1]} features")
    ensemble = StackingEnsemble(output_dir)
    ensemble.selected_features = list(feature_cols)
    ensemble.fit(X_train, y_train)
    ensemble.save(model_name=model_name)
    return ensemble, list(feature_cols)
