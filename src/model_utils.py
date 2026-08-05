import os
import pandas as pd
import numpy as np
import lightgbm as lgb

def prepare_training_data(df, feature_cols):
    df_clean = df.dropna(subset=['smp_system_price'] + feature_cols).copy()
    df_clean = df_clean[df_clean.index.year >= 2021]
    
    X = df_clean[feature_cols]
    Y = df_clean['smp_system_price']
    
    print(f"Data shape - X: {X.shape} | Y: {Y.shape}")
    return X, Y

def train_and_save_model(df, feature_cols, output_dir="outputs/models", model_name="lgb_global.txt"):
    X, Y = prepare_training_data(df, feature_cols)
    
    train_mask = X.index.year <= 2024
    val_mask   = X.index.year == 2025
    test_mask  = X.index.year == 2026
    
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val     = X[val_mask], Y[val_mask]
    
    print(f"Data split - Train: {len(X_train)} | Val: {len(X_val)}")
    
    print("Training LightGBM global model...")
    params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': 12,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=Y_train)
    val_data   = lgb.Dataset(X_val, label=Y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, model_name)
        model.save_model(model_path)
        print(f"Model saved at: {output_dir}")
        
    return model
