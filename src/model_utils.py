import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class StackingEnsemble:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.models = {}
        self.meta_learner = Ridge(alpha=1.0)
    
    def get_base_models(self):
        return {
            'lgb': lgb.LGBMRegressor(objective='mae', learning_rate=0.01, num_leaves=127, n_estimators=1500, random_state=42),
            'xgb': xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.01, max_depth=8, n_estimators=1500, random_state=42),
            'cb': CatBoostRegressor(loss_function='MAE', learning_rate=0.02, iterations=1500, depth=8, random_state=42, verbose=False),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=20))
            ])
        }
    
    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        base_models = self.get_base_models()
        oof_preds = np.zeros((len(X), len(base_models)))
        
        model_names = list(base_models.keys())
        
        print("Stage 1: Training base models (OOF predictions with TimeSeriesSplit)...")
        for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            for j, name in enumerate(model_names):
                print(f"  Fold {i+1}/5 - Training Base Model: {name}")
                model = self.get_base_models()[name]
                if name == 'lgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
                elif name == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30, verbose=False)
                
                oof_preds[val_idx, j] = model.predict(X_va)
        
        valid_idx = np.concatenate([val_idx for _, val_idx in tscv.split(X)])
        meta_X = oof_preds[valid_idx]
        meta_y = y.iloc[valid_idx]
        
        print("\nStage 2: Training Meta-Learner (Ridge Regression)...")
        self.meta_learner.fit(meta_X, meta_y)
        
        print("\nStage 3: Training base models on FULL dataset...")
        for name in model_names:
            print(f"  Training full {name}...")
            model = self.get_base_models()[name]
            model.fit(X, y)
            self.models[name] = model
            
        print("Stacking Ensemble training complete.")
        
    def predict(self, X):
        base_preds = np.column_stack([self.models[name].predict(X) for name in self.models.keys()])
        return self.meta_learner.predict(base_preds)
        
    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'stacking_ensemble.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved at: {path}")

def prepare_training_data(df, feature_cols):
    df_clean = df.dropna(subset=['smp_system_price'] + feature_cols).copy()
    df_clean = df_clean[df_clean.index.year >= 2021]
    
    X = df_clean[feature_cols]
    Y = df_clean['smp_system_price']
    
    print(f"Data shape - X: {X.shape} | Y: {Y.shape}")
    return X, Y

def train_and_save_model(df, feature_cols, output_dir="outputs/models", model_name="stacking_ensemble.pkl"):
    X, Y = prepare_training_data(df, feature_cols)
    
    train_mask = X.index.year <= 2025 # Combine Train (2021-2024) and Val (2025) into full train set since we use TimeSeriesSplit inside
    X_train, Y_train = X[train_mask], Y[train_mask]
    
    print(f"Data for Stacking - Total Train size: {len(X_train)}")
    
    ensemble = StackingEnsemble(output_dir)
    ensemble.fit(X_train, Y_train)
    ensemble.save()
        
    return ensemble
