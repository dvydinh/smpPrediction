import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from src.cycle_models import CleanCycleResidualForecaster
from src.daily_models import CINGLearForecaster, SimilarDayForecaster


def _cycle_values(index):
    return np.asarray(
        index.hour * 2 + (index.minute == 30).astype(int),
        dtype=int,
    )


class CycleRidgeStacker:
    """Fit a separate regularized blend for each half-hour cycle."""

    def __init__(self):
        self.models = {}
        self.alpha_ = np.nan
        self.coef_ = np.array([], dtype=float)
        self.cycle_coefficients_ = {}

    @staticmethod
    def _new_model(sample_count):
        splits = min(5, max(2, sample_count // 40))
        return RidgeCV(
            alphas=np.logspace(-2, 5, 20),
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=splits),
        )

    def fit(self, X, y, cycles):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cycles = np.asarray(cycles, dtype=int)
        self.models = {}
        coefficients = []
        alphas = []
        for cycle in range(48):
            mask = cycles == cycle
            if mask.sum() < 80:
                raise ValueError(f"Not enough meta samples for cycle {cycle}")
            model = self._new_model(int(mask.sum()))
            model.fit(X[mask], y[mask])
            self.models[cycle] = model
            self.cycle_coefficients_[cycle] = model.coef_.tolist()
            coefficients.append(model.coef_)
            alphas.append(model.alpha_)
        self.coef_ = np.mean(np.vstack(coefficients), axis=0)
        self.alpha_ = float(np.median(alphas))
        return self

    def predict(self, X, cycles):
        X = np.asarray(X, dtype=float)
        cycles = np.asarray(cycles, dtype=int)
        result = np.empty(len(X), dtype=float)
        for cycle, model in self.models.items():
            mask = cycles == cycle
            if mask.any():
                result[mask] = model.predict(X[mask])
        return result


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
        self.cycle_lgb_estimators_ = 1200

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
            "cycle_lgb",
        ]
        oof_predictions = np.full((len(X), len(model_names)), np.nan, dtype=float)
        cycle_iteration_history = []

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

            print("    clean cycle residual expert")
            cycle_lgb = CleanCycleResidualForecaster()
            cycle_lgb.fit(
                X_train,
                y_train,
                sample_weight=self._recency_weights(X_train.index),
                X_val=X_val,
                y_val=y_val,
                val_weight=self._recency_weights(X_val.index),
            )
            oof_predictions[val_positions, 5] = cycle_lgb.predict(X_val)
            cycle_iteration_history.extend(cycle_lgb.best_iterations_.values())

        if cycle_iteration_history:
            self.cycle_lgb_estimators_ = int(np.clip(
                np.median(cycle_iteration_history),
                100,
                1200,
            ))
            print(f"  cycle expert estimators {self.cycle_lgb_estimators_}")

        valid_positions = np.flatnonzero(np.isfinite(oof_predictions).all(axis=1))
        meta_X = oof_predictions[valid_positions]
        meta_y = y.iloc[valid_positions]
        candidate_sets = {
            "trees_global": ([0, 1, 2], "global"),
            "trees_cycle": ([0, 1, 2], "cycle"),
            "trees_cing_global": ([0, 1, 2, 3], "global"),
            "trees_cing_cycle": ([0, 1, 2, 3], "cycle"),
            "trees_similar": ([0, 1, 2, 4], "cycle"),
            "trees_cycle_lgb_global": ([0, 1, 2, 5], "global"),
            "trees_cycle_lgb_cycle": ([0, 1, 2, 5], "cycle"),
            "trees_cing_cycle_lgb": ([0, 1, 2, 3, 5], "cycle"),
            "all": ([0, 1, 2, 3, 4, 5], "cycle"),
        }
        meta_values = meta_y.to_numpy(dtype=float)
        meta_cycles = _cycle_values(X.index[valid_positions])
        clean_meta = (meta_values > 500.0) & (meta_values < 2500.0)
        selection_cut = int(len(meta_y) * 0.8)
        train_selection = clean_meta & (np.arange(len(meta_y)) < selection_cut)
        validation_selection = clean_meta & (np.arange(len(meta_y)) >= selection_cut)
        best_name = None
        best_score = np.inf
        best_validation_pred = None
        for candidate_name, (columns, meta_kind) in candidate_sets.items():
            candidate = self._new_meta_learner(meta_kind)
            self._fit_meta(
                candidate,
                meta_kind,
                meta_X[train_selection][:, columns],
                meta_values[train_selection],
                meta_cycles[train_selection],
            )
            validation_pred = self._predict_meta(
                candidate,
                meta_kind,
                meta_X[validation_selection][:, columns],
                meta_cycles[validation_selection],
            )
            validation_y = meta_values[validation_selection]
            score = np.mean(np.abs(validation_y - validation_pred))
            self.base_selection_scores_[candidate_name] = float(score)
            print(f"  candidate {candidate_name} clean mae {score:.3f}")
            if score < best_score:
                best_name = candidate_name
                best_score = score
                best_validation_pred = validation_pred

        selected_columns, self.meta_kind_ = candidate_sets[best_name]
        self.selected_candidate_ = best_name
        self.model_order_ = [model_names[column] for column in selected_columns]
        self.meta_learner = self._new_meta_learner(self.meta_kind_)
        self._fit_meta(
            self.meta_learner,
            self.meta_kind_,
            meta_X[clean_meta][:, selected_columns],
            meta_values[clean_meta],
            meta_cycles[clean_meta],
        )

        validation_cycles = meta_cycles[validation_selection]
        validation_residuals = meta_values[validation_selection] - best_validation_pred
        for cycle in range(48):
            local = validation_residuals[validation_cycles == cycle]
            if len(local):
                shrinkage = len(local) / (len(local) + 48.0)
                self.cycle_bias_[cycle] = np.clip(
                    shrinkage * np.median(local),
                    -75.0,
                    75.0,
                )

        print("Stage 2 - full-data base models")
        for name in ("lgb", "xgb", "cb"):
            model = self.get_base_models()[name]
            self.models[name] = self._fit_tree(name, model, X, y)

        if "cing_lear" in self.model_order_:
            self.models["cing_lear"] = CINGLearForecaster().fit(X, y)
        if "similar_day" in self.model_order_:
            self.models["similar_day"] = SimilarDayForecaster(neighbors=21).fit(X, y)
        if "cycle_lgb" in self.model_order_:
            self.models["cycle_lgb"] = CleanCycleResidualForecaster(
                n_estimators=self.cycle_lgb_estimators_,
            ).fit(
                X,
                y,
                sample_weight=self._recency_weights(X.index),
            )
        print(f"Selected base models - {self.model_order_}")
        print(f"Selected meta model - {self.meta_kind_}")
        print("Stacking ensemble training complete")
        return self

    @staticmethod
    def _new_meta_learner(kind="global"):
        if kind == "cycle":
            return CycleRidgeStacker()
        return RidgeCV(
            alphas=np.logspace(-2, 5, 20),
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=5),
        )

    @staticmethod
    def _fit_meta(model, kind, X, y, cycles):
        if kind == "cycle":
            return model.fit(X, y, cycles)
        return model.fit(X, y)

    @staticmethod
    def _predict_meta(model, kind, X, cycles):
        if kind == "cycle":
            return model.predict(X, cycles)
        return model.predict(X)

    def predict(self, X):
        predictions = [self.models[name].predict(X) for name in self.model_order_]
        base_predictions = np.column_stack(predictions)
        cycles = _cycle_values(X.index)
        result = self._predict_meta(
            self.meta_learner,
            self.meta_kind_,
            base_predictions,
            cycles,
        )
        return result + self.cycle_bias_[cycles]

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
