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
from src.iceemdan_utils import CEEMDANForecaster


class StackingEnsemble:
    def __init__(self, output_dir, recency_half_life_days=730):
        self.output_dir = output_dir
        self.recency_half_life_days = recency_half_life_days
        self.models = {}
        self.model_order_ = []
        self.meta_learner = None
        self.cycle_bias_ = np.zeros(48, dtype=float)
        self.base_selection_scores_ = {}

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
            "ceemdan",
        ]
        oof_predictions = np.full((len(X), len(model_names)), np.nan, dtype=float)

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

            print("    ceemdan decomposition expert")
            ceemdan = CEEMDANForecaster(n_imfs_max=6, trials=50)
            ceemdan.fit(X_train, y_train)
            oof_predictions[val_positions, 5] = ceemdan.predict(X_val)

        valid_positions = np.flatnonzero(np.isfinite(oof_predictions).all(axis=1))
        meta_X = oof_predictions[valid_positions]
        meta_y = y.iloc[valid_positions]
        candidate_sets = {
            "trees": [0, 1, 2],
            "trees_cing": [0, 1, 2, 3],
            "trees_similar": [0, 1, 2, 4],
            "trees_ceemdan": [0, 1, 2, 5],
            "trees_cing_ceemdan": [0, 1, 2, 3, 5],
            "all": [0, 1, 2, 3, 4, 5],
        }
        selection_cut = int(len(meta_y) * 0.8)
        best_name = None
        best_score = np.inf
        for candidate_name, columns in candidate_sets.items():
            candidate = self._new_meta_learner()
            candidate.fit(meta_X[:selection_cut, columns], meta_y.iloc[:selection_cut])
            validation_pred = candidate.predict(meta_X[selection_cut:, columns])
            validation_y = meta_y.iloc[selection_cut:].to_numpy()
            clean = (validation_y > 500.0) & (validation_y < 2500.0)
            score = np.mean(np.abs(validation_y[clean] - validation_pred[clean]))
            self.base_selection_scores_[candidate_name] = float(score)
            if score < best_score:
                best_name = candidate_name
                best_score = score

        selected_columns = candidate_sets[best_name]
        self.model_order_ = [model_names[column] for column in selected_columns]
        self.meta_learner = self._new_meta_learner()
        self.meta_learner.fit(meta_X[:, selected_columns], meta_y)

        meta_oof = self.meta_learner.predict(meta_X[:, selected_columns])
        residuals = meta_y.to_numpy() - meta_oof
        valid_index = X.index[valid_positions]
        cycles = valid_index.hour * 2 + (valid_index.minute == 30).astype(int)
        for cycle in range(48):
            local = residuals[cycles == cycle]
            if len(local):
                self.cycle_bias_[cycle] = np.clip(0.5 * np.median(local), -50.0, 50.0)

        print("Stage 2 - full-data base models")
        for name in ("lgb", "xgb", "cb"):
            model = self.get_base_models()[name]
            self.models[name] = self._fit_tree(name, model, X, y)

        if "cing_lear" in self.model_order_:
            self.models["cing_lear"] = CINGLearForecaster().fit(X, y)
        if "similar_day" in self.model_order_:
            self.models["similar_day"] = SimilarDayForecaster(neighbors=21).fit(X, y)
        if "ceemdan" in self.model_order_:
            self.models["ceemdan"] = CEEMDANForecaster(
                n_imfs_max=6,
                trials=50,
            ).fit(X, y)
        print(f"Selected base models - {self.model_order_}")
        print("Stacking ensemble training complete")
        return self

    @staticmethod
    def _new_meta_learner():
        return RidgeCV(
            alphas=np.logspace(-3, 4, 16),
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=5),
        )

    def predict(self, X):
        predictions = [self.models[name].predict(X) for name in self.model_order_]
        base_predictions = np.column_stack(predictions)
        result = self.meta_learner.predict(base_predictions)
        cycles = X.index.hour * 2 + (X.index.minute == 30).astype(int)
        return result + self.cycle_bias_[np.asarray(cycles, dtype=int)]

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
