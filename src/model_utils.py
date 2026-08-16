import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from src.iceemdan_utils import ICEEMDANForecaster

class StackingEnsemble:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.models = {}
        self.meta_learner = Ridge(alpha=1.0)
    def get_base_models(self):
        return {
            'lgb': lgb.LGBMRegressor(objective='mae', learning_rate=0.005, num_leaves=255, n_estimators=3000, 
                                     colsample_bytree=0.8, min_child_samples=20, random_state=42),
            'xgb': xgb.XGBRegressor(objective='reg:absoluteerror', learning_rate=0.005, max_depth=10, n_estimators=3000, 
                                    colsample_bytree=0.7, subsample=0.8, random_state=42, tree_method='hist', device='cuda'),
            'cb': CatBoostRegressor(loss_function='MAE', learning_rate=0.01, iterations=3000, depth=10, 
                                    l2_leaf_reg=3, random_state=42, verbose=False, task_type='GPU')
        }
    
    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Use raw target (log1p caused massive WMAPE degradation in Trial 7)
        y_target = y.copy()
        
        base_models = self.get_base_models()
        model_names = list(base_models.keys())
        n_models = len(model_names)
        
        oof_preds = np.zeros((len(X), n_models))  # lgb, xgb, cb (Pure Tree Stacking)
        
        print("Stage 1: Training base models (OOF predictions with TimeSeriesSplit)...")
        for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y_target.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y_target.iloc[val_idx]
            
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
        meta_y = y_target.iloc[valid_idx]
        
        print("\nStage 2: Training Meta-Learner (Ridge Regression)...")
        self.meta_learner.fit(meta_X, meta_y)
        
        print("\nStage 3: Training base models on FULL dataset...")
        for name in model_names:
            print(f"  Training full {name}...")
            model = self.get_base_models()[name]
            model.fit(X, y_target)
            self.models[name] = model
            
        print("Stacking Ensemble training complete.")
        
    def predict(self, X):
        preds_list = []
        for name, model in self.models.items():
            preds_list.append(model.predict(X))
        base_preds = np.column_stack(preds_list)
        
        # 1. Base Meta-Learner Prediction
        preds = self.meta_learner.predict(base_preds)
            
        return preds
        
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
    
    train_mask = X.index.year <= 2025
    X_train, Y_train = X[train_mask], Y[train_mask]
    
    # ==========================================================
    # Phase 0: Scout LGBM for Auto Feature Selection
    # ==========================================================
    print("=" * 60)
    print("Phase 0: Scout LGBM - Auto Feature Selection")
    print("=" * 60)
    scout = lgb.LGBMRegressor(
        objective='mae', learning_rate=0.05, num_leaves=127,
        n_estimators=500, colsample_bytree=0.8, random_state=42
    )
    scout.fit(X_train, Y_train)
    
    importances = pd.Series(scout.feature_importances_, index=feature_cols)
    importances_sorted = importances.sort_values(ascending=False)
    
    # Print full ranking
    print("\n--- Full Feature Importance Ranking ---")
    for rank, (feat, score) in enumerate(importances_sorted.items(), 1):
        print(f"  {rank:3d}. {feat:40s} {score:10.0f}")
    
    # Protected features (always keep regardless of importance)
    protected = {
        # Cyclical time encoding
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month',
        # Core price & load lags (weekly cycle)
        'smp_same_cycle_1d', 'smp_same_cycle_2d', 'smp_same_cycle_7d',
        'load_same_cycle_1d', 'load_same_cycle_2d', 'load_same_cycle_7d',
        # Calendar & regime
        'is_weekend', 'is_holiday', 'is_post_covid',
        # Morning & daily aggregates (structural)
        'morning_smp_mean', 'morning_load_mean',
        'prev_full_smp_mean', 'prev_full_gate_prob',
        # Volatility & mean-reversion (market dynamics)
        'smp_rolling_std_1d', 'smp_rolling_mean_7d',
        # Core physics
        'residual_load_proxy',
    }
    
    # Cut bottom 20% by importance (unless protected)
    threshold = importances.quantile(0.20)
    selected_for_analysis = [f for f in feature_cols if f in protected or importances[f] >= threshold]
    removed = [f for f in feature_cols if f not in selected_for_analysis]
    
    print(f"\n--- Selection Summary (ANALYSIS ONLY) ---")
    print(f"  Threshold (20th pct): {threshold:.0f}")
    print(f"  Total: {len(feature_cols)} -> Would Select: {len(selected_for_analysis)} | Would Remove: {len(removed)}")
    if removed:
        print(f"  Would Remove: {removed}")
    print("=" * 60)
    
    # CRITICAL FIX (Trial 17 Analysis): 
    # Dropping features based on LGBM's perspective cripples XGBoost and CatBoost.
    # We must train the ensemble on ALL features to preserve model diversity.
    selected = feature_cols
    
    # ==========================================================
    # Phase 1-3: Full Stacking Ensemble on ALL features
    # ==========================================================
    X_train_sel = X_train[selected]
    
    print(f"\nData for Stacking - Train: {len(X_train_sel)} x {X_train_sel.shape[1]} features")
    
    ensemble = StackingEnsemble(output_dir)
    ensemble.selected_features = selected
    ensemble.fit(X_train_sel, Y_train)
    ensemble.save()
        
    return ensemble, selected
