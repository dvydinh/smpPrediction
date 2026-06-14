import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit


class StackingEnsemble:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.models = {}
        self.model_order_ = []
        self.meta_learner = None

    def get_base_models(self):
        return {
            "lgb": lgb.LGBMRegressor(
                objective="mae",
                learning_rate=0.01,
                num_leaves=127,
                n_estimators=1500,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
            ),
            "xgb": xgb.XGBRegressor(
                objective="reg:absoluteerror",
                learning_rate=0.01,
                max_depth=8,
                n_estimators=1500,
                colsample_bytree=0.7,
                random_state=42,
                tree_method="hist",
            ),
            "cb": CatBoostRegressor(
                loss_function="MAE",
                learning_rate=0.02,
                iterations=1500,
                depth=8,
                random_seed=42,
                verbose=False,
            ),
        }

    def fit(self, X, y):
        X = X.sort_index()
        y = y.loc[X.index]
        tscv = TimeSeriesSplit(n_splits=5)
        model_names = list(self.get_base_models().keys())
        oof = np.full((len(X), len(model_names)), np.nan)

        print("Stage 1 - generating out-of-fold predictions")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            print(f"  fold {fold}/5")
            for col, (name, model) in enumerate(self.get_base_models().items()):
                model.fit(X_train, y_train)
                oof[val_idx, col] = model.predict(X_val)

        valid = np.isfinite(oof).all(axis=1)
        self.meta_learner = RidgeCV(
            alphas=np.logspace(-2, 5, 20),
            scoring="neg_mean_absolute_error",
        )
        self.meta_learner.fit(oof[valid], y.iloc[valid].to_numpy(dtype=float))

        print("Stage 2 - full data refit")
        self.model_order_ = model_names
        for name, model in self.get_base_models().items():
            model.fit(X, y)
            self.models[name] = model

        print("Training complete")
        return self

    def predict(self, X):
        preds = np.column_stack([
            self.models[name].predict(X) for name in self.model_order_
        ])
        return self.meta_learner.predict(preds)

    def save(self, model_name="stacking_ensemble.pkl"):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, model_name)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved at {path}")
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


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
