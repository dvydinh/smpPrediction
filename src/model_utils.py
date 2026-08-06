import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from src.iceemdan_utils import ICEEMDANForecaster

class StackingEnsemble:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.models = {}
        self.meta_learner = Ridge(alpha=1.0)
    
    def get_base_models(self):
        return {
            'lgb': lgb.LGBMRegressor(objective='mae', learning_rate=0.005, num_leaves=255, n_estimators=3000, colsample_bytree=0.8, min_child_samples=20, random_state=42),
            'xgb': xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.005, max_depth=10, n_estimators=3000, colsample_bytree=0.8, random_state=42),
            'cb': CatBoostRegressor(loss_function='MAE', learning_rate=0.01, iterations=3000, depth=10, l2_leaf_reg=3, random_state=42, verbose=False)
        }
    
    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Log-transform target to handle right-skewed price distribution
        y_log = np.log1p(y)
        
        oof_preds = np.zeros((len(X), 4))  # lgb, xgb, cb, iceemdan
        
        base_models = self.get_base_models()
        model_names = list(base_models.keys())
        
        print("Stage 1: Training base models (OOF predictions with TimeSeriesSplit)...")
        for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
            
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
            
            # ICEEMDAN base model (4th column)
            # CRITICAL: decompose ONLY on training target to prevent leakage
            print(f"  Fold {i+1}/5 - Training Base Model: iceemdan")
            iceemdan_model = ICEEMDANForecaster(n_imfs_max=6)
            iceemdan_model.fit(X_tr, y_tr)
            oof_preds[val_idx, 3] = iceemdan_model.predict(X_va)
        
        valid_idx = np.concatenate([val_idx for _, val_idx in tscv.split(X)])
        meta_X = oof_preds[valid_idx]
        meta_y = y_log.iloc[valid_idx]
        
        print("\nStage 2: Training Meta-Learner (Ridge Regression)...")
        self.meta_learner.fit(meta_X, meta_y)
        
        print("\nStage 3: Training base models on FULL dataset...")
        for name in model_names:
            print(f"  Training full {name}...")
            model = self.get_base_models()[name]
            model.fit(X, y_log)
            self.models[name] = model
        
        # Train ICEEMDAN on full dataset
        print("  Training full iceemdan...")
        self.iceemdan_model = ICEEMDANForecaster(n_imfs_max=6)
        self.iceemdan_model.fit(X, y_log)
        self.models['iceemdan'] = self.iceemdan_model
            
        print("Stacking Ensemble training complete.")
        
    def predict(self, X):
        preds_list = []
        for name, model in self.models.items():
            preds_list.append(model.predict(X))
        base_preds = np.column_stack(preds_list)
        log_preds = self.meta_learner.predict(base_preds)
        return np.expm1(log_preds)  # Inverse log-transform back to VND
        
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
