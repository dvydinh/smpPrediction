import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

LGB_PARAMS = {
    'objective': 'mae', 'metric': 'mae', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.005, 'feature_fraction': 0.5,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 20, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 200
CYCLES_PER_DAY = 48
PRICE_CAP = 1778.6

def prepare_daily_matrices(df, target_col='smp_system_price'):
    feature_cols = [c for c in df.columns if c not in [target_col, 'cycle_id', '_date']]
    df['_date'] = pd.to_datetime(df.index.date)
    dates = sorted(df['_date'].unique())
    X_list, Y_list, date_list = [], [], []

    for i in range(len(dates) - 1):
        snapshot = pd.Timestamp(dates[i]) + pd.Timedelta(hours=7, minutes=30)
        if snapshot not in df.index: continue
        d1_start = pd.Timestamp(dates[i + 1])
        targets = df.loc[d1_start:d1_start + pd.Timedelta(hours=23, minutes=30), target_col]
        
        if len(targets) != CYCLES_PER_DAY: continue
        feat_row = df.loc[snapshot, feature_cols]
        if feat_row.isna().all() or targets.isna().any(): continue
        
        X_list.append(feat_row.values)
        Y_list.append(targets.values)
        date_list.append(dates[i + 1])

    X_daily = np.array(X_list, dtype=np.float64)
    Y_daily = np.array(Y_list, dtype=np.float64)
    dates_arr = np.array(date_list)

    nan_mask = np.isnan(X_daily)
    if nan_mask.any():
        medians = np.nanmedian(X_daily, axis=0)
        for j in range(X_daily.shape[1]):
            X_daily[np.isnan(X_daily[:, j]), j] = medians[j]

    print(f'[INFO] X_daily_shape: {X_daily.shape} | Y: {Y_daily.shape}')
    return X_daily, Y_daily, dates_arr, feature_cols

def create_median_baseline(X_daily, Y_daily, dates_arr):
    MIN_WEEKS = 21
    Y_base = np.full_like(Y_daily, np.nan)
    for i in range(MIN_WEEKS, len(Y_daily)):
        candidates = []
        for w in [7, 14, 21]:
            if i - w >= 0:
                candidates.append(Y_daily[i - w])
        Y_base[i] = np.median(candidates, axis=0)

    X_daily = X_daily[MIN_WEEKS:]
    Y_daily = Y_daily[MIN_WEEKS:]
    dates_arr = dates_arr[MIN_WEEKS:]
    Y_base = Y_base[MIN_WEEKS:]
    
    Y_res = np.log1p(Y_daily) - np.log1p(Y_base)
    print(f'[INFO] Residuals_ready_shape: {X_daily.shape}')
    return X_daily, Y_daily, Y_base, Y_res, dates_arr

def flatten_for_global_model(X_daily, Y_daily, Y_base, Y_res, dates_arr, feature_cols):
    N_days, F = X_daily.shape
    _, C = Y_daily.shape

    X_rep = np.repeat(X_daily, C, axis=0)

    cycle_ids = np.tile(np.arange(C, dtype=np.float32), N_days)
    sin_h = np.sin(2 * np.pi * cycle_ids / 48.0)
    cos_h = np.cos(2 * np.pi * cycle_ids / 48.0)

    X_flat = np.hstack([X_rep, cycle_ids.reshape(-1, 1), sin_h.reshape(-1, 1), cos_h.reshape(-1, 1)]).astype(np.float32)
    Y_res_flat = Y_res.flatten().astype(np.float32)
    Y_base_flat = Y_base.flatten().astype(np.float32)
    Y_daily_flat = Y_daily.flatten().astype(np.float32)
    dates_flat = np.repeat(dates_arr, C)
    
    extended_feature_cols = feature_cols + ['target_cycle_id', 'target_sin_hour', 'target_cos_hour']
    print(f'[INFO] X_flat_shape: {X_flat.shape}')
    
    return X_flat, Y_daily_flat, Y_base_flat, Y_res_flat, dates_flat, extended_feature_cols

def split_and_train_lgb(X_flat, Y_res_flat, Y_base_flat, dates_flat, ext_feature_cols, output_dir=None):
    dates_pd = pd.to_datetime(dates_flat)
    train_mask = dates_pd <= '2026-03-31'
    test_mask  = dates_pd >= '2026-04-01'

    X_train, Y_train = X_flat[train_mask], Y_res_flat[train_mask]
    X_test,  Y_test, Y_base_test = X_flat[test_mask], Y_res_flat[test_mask], Y_base_flat[test_mask]

    n_val = max(30, int(len(X_train) * 0.1))
    X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]
    X_vl, Y_vl = X_train[-n_val:], Y_train[-n_val:]
    
    print(f'[INFO] Data_split: Train= {len(X_tr)} | Val: {n_val} | Test: {len(X_test)}')

    train_ds = lgb.Dataset(X_tr, label=Y_tr, feature_name=ext_feature_cols, free_raw_data=True)
    val_ds   = lgb.Dataset(X_vl, label=Y_vl, reference=train_ds, free_raw_data=True)

    print('[INFO] Training LightGBM Global Model...')
    global_model = lgb.train(
        LGB_PARAMS, train_ds, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(10)],
    )

    val_mae = global_model.best_score['valid_0']['l1']
    print(f'[INFO] Training_complete. Val_MAE: {val_mae:.4f}')
    
    if output_dir:
        import os
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lgb_global.txt')
        global_model.save_model(model_path)
        print(f'[INFO] Model_saved: {model_path}')
        
    return global_model, X_test, Y_test, Y_base_test

