import os, glob, gc, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

KAGGLE_INPUT = Path('/kaggle/input')
if KAGGLE_INPUT.exists():
    found_market = list(KAGGLE_INPUT.rglob("market/smp_prices_nsmo.csv"))
    DATA_ROOT = found_market[0].parent.parent if found_market else (list(KAGGLE_INPUT.iterdir())[0] if list(KAGGLE_INPUT.iterdir()) else Path("."))
    print(f'Kaggle dataset: {DATA_ROOT}')
else:
    DATA_ROOT = Path('data/raw')
    print(f'Local: {DATA_ROOT}')

MARKET_DIR = DATA_ROOT / 'market'
HYDRO_DIR  = DATA_ROOT / 'hydro'
EXOG_DIR   = DATA_ROOT / 'exogenous'
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_CAP = 1778.6
NEAR_ZERO = 2.0
CYCLES_PER_DAY = 48
LAG_SHIFT = 48

LGB_PARAMS = {
    'objective': 'mae', 'metric': 'mae', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.005, 'feature_fraction': 0.5,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 20, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 200


master_idx = pd.date_range('2021-01-01', '2026-06-19 23:30', freq='30min')
df = pd.DataFrame(index=master_idx)
df.index.name = 'datetime'

smp = pd.read_csv(MARKET_DIR / 'smp_prices_nsmo.csv')
smp['datetime'] = pd.to_datetime(smp['datetime'])
smp = smp.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
for col in ['smp_system_price', 'smp_north_price', 'smp_central_price', 'smp_south_price']:
    df[col] = smp[col]

df['cycle_id'] = df.index.hour * 2 + df.index.minute // 30

load = pd.read_csv(MARKET_DIR / 'load_data_nsmo.csv')
load['datetime'] = pd.to_datetime(load['datetime'])
load = load.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
for col in ['load_total_mw', 'load_north_mw', 'load_central_mw', 'load_south_mw']:
    df[col] = load[col]

print(f'SMP: {len(smp)} rows | Load: {len(load)} rows')
del smp, load; gc.collect()

hydro_files = sorted(glob.glob(str(HYDRO_DIR / 'hydro_hourly_*.csv')))
print(f'{len(hydro_files)} hydro files')

hydro_chunks = []
for f in hydro_files:
    h = pd.read_csv(f)
    h['datetime'] = pd.to_datetime(h['datetime'])
    agg = h.groupby('datetime').agg({
        'inflow_m3s': 'sum', 'total_discharge_m3s': 'sum',
        'plant_discharge_m3s': 'sum', 'spill_discharge_m3s': 'sum',
        'water_level_m': 'mean',
    })
    hydro_chunks.append(agg)

hydro = pd.concat(hydro_chunks).groupby(level=0).mean()
hydro.columns = ['hydro_' + c for c in hydro.columns]
hydro = hydro.sort_index()
idx_30 = pd.date_range(hydro.index.min(), hydro.index.max() + pd.Timedelta('30min'), freq='30min')
hydro = hydro.reindex(idx_30).ffill()
for col in hydro.columns:
    df[col] = hydro[col]
print(f'Hydro aggregated: {len(hydro)} rows')
del hydro_chunks, hydro; gc.collect()

disp = pd.read_csv(MARKET_DIR / 'dispatch_capacity_nsmo.csv', encoding='utf-8')
disp['date'] = pd.to_datetime(disp['date'])
patterns = {'total': 'quốc', 'hydro': 'Thủy', 'solar': 'trời trang', 'wind': 'gió'}
disp_frames = []
for label, pat in patterns.items():
    sub = disp[disp['resource_type'].str.contains(pat, case=False, na=False)].set_index('date')
    renamed = sub[['installed_capacity_mw', 'expected_midday_low_load_mw', 'expected_evening_peak_mw']]
    renamed.columns = [f'disp_{label}_installed_mw', f'disp_{label}_midday_mw', f'disp_{label}_evening_mw']
    disp_frames.append(renamed)
disp_pivot = pd.concat(disp_frames, axis=1)
disp_pivot.index = pd.to_datetime(disp_pivot.index)

dates_ts = pd.to_datetime(df.index.date)
for col in disp_pivot.columns:
    df[col] = dates_ts.map(disp_pivot[col].to_dict()).values
print(f'Dispatch: {len(disp_pivot)} dates')
del disp, disp_pivot; gc.collect()

weather = pd.read_csv(EXOG_DIR / 'weather_3_regions_30min_open_meteo.csv')
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather = weather.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
drop_cols = [c for c in ('date', 'cycle_id', 'source', 'timezone') if c in weather.columns]
weather = weather.drop(columns=drop_cols)
weather_cols = [c for c in weather.columns if weather[c].dtype in (np.float64, np.int64, float, int)]
for col in weather_cols:
    df[col] = weather[col]
print(f'Weather: {len(weather_cols)} features')
del weather; gc.collect()

fuel = pd.read_csv(EXOG_DIR / 'fuel_macro_yfinance_clean.csv')
fuel['date'] = pd.to_datetime(fuel['date'])
fuel = fuel.sort_values('date').drop_duplicates('date').set_index('date')
fuel_cols = [c for c in ['coal_proxy_price', 'brent_price', 'gas_proxy_price', 'usd_vnd', 'dxy_index'] if c in fuel.columns]
for col in fuel_cols:
    df[col] = dates_ts.map(fuel[col].to_dict()).values
print(f'Fuel: {len(fuel_cols)} features')
del fuel; gc.collect()

cal = pd.read_csv(EXOG_DIR / 'calendar_vietnam.csv')
cal['date'] = pd.to_datetime(cal['date'])
cal = cal.set_index('date')
for col in ['is_weekend', 'is_workday', 'is_holiday', 'is_tet', 'is_pre_holiday', 'is_post_holiday', 'season']:
    if col in cal.columns:
        df[col] = dates_ts.map(cal[col].to_dict()).values
del cal; gc.collect()

print(f'\nMerged shape: {df.shape}')

# --- MISSING VALUE HANDLING (NO INTERPOLATION - ffill only) ---
for col in df.select_dtypes(include=[np.number]).columns:
    if not df[col].isna().any(): continue
    if df[col].isna().any(): df[col] = df[col].fillna(df[col].shift(336))
    if df[col].isna().any(): df[col] = df[col].ffill()
    if df[col].isna().any(): df[col] = df[col].bfill()

hour_frac = df.index.hour + df.index.minute / 60.0
df['sin_hour']  = np.sin(2 * np.pi * hour_frac / 24)
df['cos_hour']  = np.cos(2 * np.pi * hour_frac / 24)
df['sin_dow']   = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['cos_dow']   = np.cos(2 * np.pi * df.index.dayofweek / 7)
df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)
df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)

for lag in [48, 49, 50, 96, 336]:
    df[f'smp_lag_{lag}']  = df['smp_system_price'].shift(lag)
    df[f'load_lag_{lag}'] = df['load_total_mw'].shift(lag)

# --- DENSE LAG VECTOR: 16 recent SMP values (07:00 back to 23:30) ---
for offset in range(1, 17):
    df[f'smp_recent_{offset}'] = df['smp_system_price'].shift(offset)

for lag in [48, 336]:
    df[f'smp_north_lag_{lag}'] = df['smp_north_price'].shift(lag)
    df[f'smp_south_lag_{lag}'] = df['smp_south_price'].shift(lag)

smp_shifted  = df['smp_system_price'].shift(LAG_SHIFT)
load_shifted = df['load_total_mw'].shift(LAG_SHIFT)

for label, window in {'24h': 48, '72h': 144}.items():
    df[f'smp_rolling_mean_{label}']  = smp_shifted.rolling(window, min_periods=window//2).mean()
    df[f'smp_rolling_std_{label}']   = smp_shifted.rolling(window, min_periods=window//2).std()
    df[f'load_rolling_mean_{label}'] = load_shifted.rolling(window, min_periods=window//2).mean()
    df[f'load_rolling_std_{label}']  = load_shifted.rolling(window, min_periods=window//2).std()

df['price_spread_ns_lag48'] = df['smp_north_price'].shift(LAG_SHIFT) - df['smp_south_price'].shift(LAG_SHIFT)
df['load_ramp_lag48'] = df['load_total_mw'].shift(LAG_SHIFT) - df['load_total_mw'].shift(LAG_SHIFT + 1)
df['smp_yesterday_mean']  = smp_shifted.rolling(CYCLES_PER_DAY).mean()
df['smp_yesterday_max']   = smp_shifted.rolling(CYCLES_PER_DAY).max()
df['smp_yesterday_min']   = smp_shifted.rolling(CYCLES_PER_DAY).min()
df['smp_yesterday_zero_ratio'] = (df['smp_system_price'].shift(LAG_SHIFT) <= NEAR_ZERO).astype(float).rolling(CYCLES_PER_DAY).mean()

if 'hydro_total_discharge_m3s' in df.columns:
    df['hydro_discharge_rolling_24h'] = df['hydro_total_discharge_m3s'].shift(LAG_SHIFT).rolling(48, min_periods=24).mean()

df = df.drop(columns=['smp_north_price', 'smp_central_price', 'smp_south_price'], errors='ignore')

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any(): df[col] = df[col].ffill().bfill()


# --- PHYSICS-INFORMED FEATURES ---
if 'shortwave_radiation_hcmc' in df.columns and 'disp_solar_installed_mw' in df.columns:
    df['solar_gen_proxy'] = (df['shortwave_radiation_hcmc'] / 1000.0) * df['disp_solar_installed_mw'] * 0.75
else:
    df['solar_gen_proxy'] = 0

if 'wind_speed_hcmc' in df.columns and 'disp_wind_installed_mw' in df.columns:
    df['wind_gen_proxy'] = (df['wind_speed_hcmc'] / 10.0) * df['disp_wind_installed_mw'] * 0.3
else:
    df['wind_gen_proxy'] = 0

if 'load_total_mw' in df.columns and 'solar_gen_proxy' in df.columns:
    df['residual_load_proxy'] = df['load_total_mw'].shift(LAG_SHIFT) - df['solar_gen_proxy'].shift(LAG_SHIFT) - df['wind_gen_proxy'].shift(LAG_SHIFT)

if 'disp_total_installed_mw' in df.columns and 'residual_load_proxy' in df.columns:
    hydro_cap = df['disp_hydro_installed_mw'].fillna(0) if 'disp_hydro_installed_mw' in df.columns else 0
    solar_cap = df['disp_solar_installed_mw'].fillna(0) if 'disp_solar_installed_mw' in df.columns else 0
    wind_cap = df['disp_wind_installed_mw'].fillna(0) if 'disp_wind_installed_mw' in df.columns else 0
    thermal_cap = df['disp_total_installed_mw'] - hydro_cap - solar_cap - wind_cap
    df['thermal_margin_proxy'] = thermal_cap - df['residual_load_proxy']

if 'hydro_water_level_m' in df.columns:
    df['hydro_depletion_index'] = df['hydro_water_level_m'].shift(LAG_SHIFT)

    if 'thermal_margin_proxy' in df.columns and 'temperature_danang' in df.columns:
        df['margin_temp_interaction'] = df['thermal_margin_proxy'] * df['temperature_danang'].shift(LAG_SHIFT)

    # Probabilistic classification features (Spike and Zero proxy)
    df['is_spike'] = (df['smp_system_price'] >= 1500).astype(int)
    df['is_zero'] = (df['smp_system_price'] <= 100).astype(int)
    df['spike_prob_24h'] = df['is_spike'].shift(LAG_SHIFT).rolling(48).mean()
    df['spike_prob_72h'] = df['is_spike'].shift(LAG_SHIFT).rolling(144).mean()
    df['zero_prob_24h'] = df['is_zero'].shift(LAG_SHIFT).rolling(48).mean()
    df['zero_prob_72h'] = df['is_zero'].shift(LAG_SHIFT).rolling(144).mean()

print(f'Final shape: {df.shape}, NaN: {df.isna().sum().sum()}')



TARGET = 'smp_system_price'
feature_cols = [c for c in df.columns if c not in [TARGET, 'cycle_id']]

df['_date'] = pd.to_datetime(df.index.date)
dates = sorted(df['_date'].unique())
X_list, Y_list, date_list = [], [], []

for i in range(len(dates) - 1):
    snapshot = pd.Timestamp(dates[i]) + pd.Timedelta(hours=7, minutes=30)
    if snapshot not in df.index: continue
    d1_start = pd.Timestamp(dates[i + 1])
    targets = df.loc[d1_start:d1_start + pd.Timedelta(hours=23, minutes=30), TARGET]
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

print(f'X: {X_daily.shape} | Y: {Y_daily.shape} | {dates_arr[0]} to {dates_arr[-1]}')
del df; gc.collect()

naive_preds = np.full_like(Y_daily, np.nan)
for i in range(7, len(Y_daily)):
    naive_preds[i] = Y_daily[i - 7]

valid = ~np.isnan(naive_preds).any(axis=1)
naive_rmse = np.sqrt(mean_squared_error(Y_daily[valid].flatten(), naive_preds[valid].flatten()))
naive_mae  = mean_absolute_error(Y_daily[valid].flatten(), naive_preds[valid].flatten())
print(f'Naive lag-7 RMSE: {naive_rmse:.2f} | MAE: {naive_mae:.2f}')

# --- RESIDUAL LEARNING SETUP (MEDIAN BASELINE) ---
# Use median of same-weekday prices from past 3 weeks (D-7, D-14, D-21)
# This is more robust than a single D-7 which can be corrupted by spikes/trips
MIN_WEEKS = 21  # need at least 21 days of history
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
print(f'Residuals prepared (median baseline). X: {X_daily.shape}')



# --- VERTICAL FLATTENING FOR GLOBAL MODEL (VECTORIZED) ---
N_days, F = X_daily.shape
_, C = Y_daily.shape

# repeat each day's features 48 times
X_rep = np.repeat(X_daily, C, axis=0)  # (N_days*48, F)

# cycle ids and trigonometric encodings
cycle_ids = np.tile(np.arange(C, dtype=np.float32), N_days)  # (N_days*48,)
sin_h = np.sin(2 * np.pi * cycle_ids / 48.0)
cos_h = np.cos(2 * np.pi * cycle_ids / 48.0)

X_daily = np.hstack([X_rep, cycle_ids.reshape(-1, 1), sin_h.reshape(-1, 1), cos_h.reshape(-1, 1)]).astype(np.float32)
Y_res = Y_res.flatten().astype(np.float32)
Y_base = Y_base.flatten().astype(np.float32)
Y_daily = Y_daily.flatten().astype(np.float32)
dates_arr = np.repeat(dates_arr, C)

del X_rep, cycle_ids, sin_h, cos_h; gc.collect()

feature_cols = feature_cols + ['target_cycle_id', 'target_sin_hour', 'target_cos_hour']
print(f'Flattened X: {X_daily.shape}, Y_res: {Y_res.shape}')



dates_pd = pd.to_datetime(dates_arr)
train_mask = dates_pd <= '2026-03-31'
test_mask  = dates_pd >= '2026-04-01'

X_train, Y_train, Y_base_train = X_daily[train_mask], Y_res[train_mask], Y_base[train_mask]
X_test,  Y_test,  Y_base_test  = X_daily[test_mask],  Y_res[test_mask],  Y_base[test_mask]

n_val = max(30, int(len(X_train) * 0.1))
X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]
Y_base_vl = Y_base_train[-n_val:]
X_vl, Y_vl = X_train[-n_val:], Y_train[-n_val:]
print(f'Train: {len(X_tr)} | Val: {n_val} | Test: {len(X_test)}')

%%time
# SINGLE GLOBAL MODEL
train_ds = lgb.Dataset(X_tr, label=Y_tr, feature_name=feature_cols, free_raw_data=True)
val_ds   = lgb.Dataset(X_vl, label=Y_vl, reference=train_ds, free_raw_data=True)

global_model = lgb.train(
    LGB_PARAMS, train_ds, num_boost_round=NUM_BOOST_ROUND,
    valid_sets=[val_ds],
    callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(10)],
)

val_mae = global_model.best_score['valid_0']['l1']
print(f'\nDone. Global model trained. Val MAE: {val_mae:.4f}')


Y_pred_res_flat = global_model.predict(X_test)
Y_pred_log_flat = np.log1p(Y_base_test) + Y_pred_res_flat
Y_pred_flat = np.clip(np.expm1(Y_pred_log_flat), 0, PRICE_CAP)
Y_test_actual_flat = Y_test + Y_base_test

# Reshape back to compute cycle-specific metrics properly
N_test_days = len(Y_pred_flat) // CYCLES_PER_DAY
Y_pred = Y_pred_flat.reshape(N_test_days, CYCLES_PER_DAY)
Y_test_actual = Y_test_actual_flat.reshape(N_test_days, CYCLES_PER_DAY)

test_rmse = np.sqrt(mean_squared_error(Y_test_actual_flat, Y_pred_flat))
test_mae  = mean_absolute_error(Y_test_actual_flat, Y_pred_flat)

print(f'Test RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}')
print(f'Naive RMSE: {naive_rmse:.2f} | MAE: {naive_mae:.2f}')
print(f'Improvement: {(1 - test_rmse/naive_rmse)*100:.1f}% RMSE, {(1 - test_mae/naive_mae)*100:.1f}% MAE')

cycle_rmses, cycle_maes = [], []
for k in range(CYCLES_PER_DAY):
    cycle_rmses.append(np.sqrt(mean_squared_error(Y_test_actual[:, k], Y_pred[:, k])))
    cycle_maes.append(mean_absolute_error(Y_test_actual[:, k], Y_pred[:, k]))


test_dates_unique = np.unique(dates_arr[test_mask])
n_sample = min(6, len(test_dates_unique))
sample_idx = np.linspace(0, len(test_dates_unique) - 1, n_sample, dtype=int)
tick_pos = range(0, 48, 4)
tick_labels = [f'{c//2:02d}:{(c%2)*30:02d}' for c in tick_pos]

fig, axes = plt.subplots(n_sample, 1, figsize=(16, 4 * n_sample))
if n_sample == 1: axes = [axes]
for ax, i in zip(axes, sample_idx):
    ax.plot(range(48), Y_test_actual[i], 'b-o', ms=3, lw=1.5, label='Actual', alpha=0.8)
    ax.plot(range(48), Y_pred[i], 'r--s', ms=3, lw=1.5, label='Predicted', alpha=0.8)
    ax.set_title(pd.to_datetime(test_dates_unique[i]).strftime('%Y-%m-%d (%A)'), fontweight='bold')
    ax.set(ylabel='SMP (VND)'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()


residuals = Y_test_actual_flat - Y_pred_flat
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
axes[0].hist(residuals, bins=100, color='purple', alpha=0.7)
axes[0].set_title('Residuals Distribution (Actual - Predicted)')
axes[0].grid(alpha=0.3)

axes[1].plot(Y_test_actual_flat, Y_pred_flat, 'o', color='purple', alpha=0.3, ms=2)
axes[1].plot([0, PRICE_CAP], [0, PRICE_CAP], 'r--', lw=2)
axes[1].set(xlabel='Actual', ylabel='Predicted', title='Actual vs Predicted')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Residuals: mean={residuals.mean():.2f}, std={residuals.std():.2f}')


gain = global_model.feature_importance(importance_type='gain')

ranking = pd.DataFrame({'feature': feature_cols, 'avg_gain': gain})
ranking = ranking.sort_values('avg_gain', ascending=False).reset_index(drop=True)

top = ranking.head(20)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(20), top['avg_gain'].values[::-1], color='steelblue', alpha=0.8)
ax.set_yticks(range(20)); ax.set_yticklabels(top['feature'].values[::-1], fontsize=9)
ax.set_title('Top 20 Features (avg gain)', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(ranking.head(20).to_string(index=False))


cv_splits = [
    {'name': 'Fold 1', 'train_end': '2024-06-30', 'val_start': '2024-07-01', 'val_end': '2025-03-31'},
    {'name': 'Fold 2', 'train_end': '2025-06-30', 'val_start': '2025-07-01', 'val_end': '2026-03-31'},
    {'name': 'Final',  'train_end': '2026-03-31', 'val_start': '2026-04-01', 'val_end': '2026-06-19'},
]

cv_results = []
for split in cv_splits:
    tr_mask = dates_pd <= split['train_end']
    vl_mask = (dates_pd >= split['val_start']) & (dates_pd <= split['val_end'])
    if vl_mask.sum() == 0: continue
    Xtr, Ytr = X_daily[tr_mask], Y_res[tr_mask]
    Xvl, Yvl = X_daily[vl_mask], Y_res[vl_mask]
    Ybase_vl = Y_base[vl_mask]
    nv = max(20 * 48, int(len(Xtr) * 0.1))
    print(f'{split["name"]}: train={tr_mask.sum()//48}, val={vl_mask.sum()//48} days')
    
    td = lgb.Dataset(Xtr[:-nv], label=Ytr[:-nv], feature_name=feature_cols, free_raw_data=True)
    vd = lgb.Dataset(Xtr[-nv:], label=Ytr[-nv:], reference=td, free_raw_data=True)
    m = lgb.train(LGB_PARAMS, td, num_boost_round=NUM_BOOST_ROUND,
                  valid_sets=[vd], callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)])
    Ypred_cv_res_flat = m.predict(Xvl)
    del td, vd, m; gc.collect()
    
    Ypred_cv_log = np.log1p(Ybase_vl) + Ypred_cv_res_flat
    Ypred_cv_flat = np.clip(np.expm1(Ypred_cv_log), 0, PRICE_CAP)
    Yvl_actual_flat = Yvl + Ybase_vl
    
    rmse = np.sqrt(mean_squared_error(Yvl_actual_flat, Ypred_cv_flat))
    mae  = mean_absolute_error(Yvl_actual_flat, Ypred_cv_flat)
    cv_results.append({'fold': split['name'], 'rmse': rmse, 'mae': mae})
    print(f'  RMSE: {rmse:.2f} | MAE: {mae:.2f}\n')


model_dir = OUTPUT_DIR / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

global_model.save_model(str(model_dir / 'lgb_global.txt'))

metadata = {
    'feature_names': feature_cols,
    'params': LGB_PARAMS,
    'num_boost_round': NUM_BOOST_ROUND,
    'test_rmse': float(test_rmse), 'test_mae': float(test_mae),
    'naive_rmse': float(naive_rmse), 'naive_mae': float(naive_mae),
    'cv_results': cv_results,
    'cycle_rmses': [float(r) for r in cycle_rmses],
    'cycle_maes': [float(m) for m in cycle_maes],
}
with open(model_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Saved global model to {model_dir}')
for f in sorted(OUTPUT_DIR.rglob('*')):
    if f.is_file(): print(f'  {f.relative_to(OUTPUT_DIR)} ({f.stat().st_size / 1024:.0f} KB)')


print(f'Data: {len(dates_arr)} days, {len(feature_cols)} features')
print(f'Model: 1x Global LightGBM (MAE, {LGB_PARAMS["num_leaves"]} leaves)\n')
print(f'Test (2026-04 to 2026-06):')
print(f'  LightGBM  RMSE={test_rmse:.2f}  MAE={test_mae:.2f}')
print(f'  Naive     RMSE={naive_rmse:.2f}  MAE={naive_mae:.2f}')
print(f'  Gain: {(1-test_rmse/naive_rmse)*100:+.1f}% RMSE, {(1-test_mae/naive_mae)*100:+.1f}% MAE\n')
for r in cv_results:
    print(f'  {r["fold"]}: RMSE={r["rmse"]:.2f} MAE={r["mae"]:.2f}')

# --- FAIR EVALUATION: EXCLUDING OUTLIERS ---
# The market contains extreme outliers (price cap at 1778.6 and near-zero prices).
# We evaluate the MAPE and MAE strictly on normal market conditions (prices between 100 and 1778).

print('\n' + '='*50)
print('FINAL FAIR EVALUATION (EXCLUDING OUTLIERS)')
print('='*50)

mask_outliers = (Y_test_actual > 100) & (Y_test_actual < 1778.0)
y_true_clean = Y_test_actual[mask_outliers]
y_pred_clean = Y_pred[mask_outliers]

mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
rmse_clean = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

print(f'Valid evaluation samples: {len(y_true_clean)} / {Y_test_actual.size} ({(len(y_true_clean)/Y_test_actual.size)*100:.1f}%)')
print(f'MAE (Clean): {mae_clean:.2f} VND')
print(f'RMSE (Clean): {rmse_clean:.2f} VND')
print(f'MAPE (Clean): {mape:.2f}%')
print('='*50 + '\n')


from kaggle_secrets import UserSecretsClient
import os, shutil
from datetime import datetime

try:
    user_secrets = UserSecretsClient()
    github_token = user_secrets.get_secret('GITHUB_TOKEN')
    
    repo_url = f'https://dvydinh:{github_token}@github.com/dvydinh/smpPrediction.git'
    clone_dir = '/kaggle/temp_repo'
    
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    os.system(f'git clone {repo_url} {clone_dir}')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f'{clone_dir}/outputs/kaggle_runs/run_{timestamp}'
    os.makedirs(f'{run_dir}/models', exist_ok=True)
    
    for f in os.listdir('/kaggle/working/models'):
        shutil.copy2(f'/kaggle/working/models/{f}', f'{run_dir}/models/{f}')
        
    for f in ['actual_vs_predicted.png', 'residual_analysis.png', 'feature_importance.png']:
        if os.path.exists(f'/kaggle/working/{f}'):
            shutil.copy2(f'/kaggle/working/{f}', f'{run_dir}/{f}')
            
    os.system(f'cd {clone_dir} && git config user.email "doanvy.dinh27@gmail.com"')
    os.system(f'cd {clone_dir} && git config user.name "dvydinh"')
    os.system(f'cd {clone_dir} && git add outputs/kaggle_runs/')
    os.system(f'cd {clone_dir} && git commit -m "save model output {timestamp}"')
    os.system(f'cd {clone_dir} && git push')
    
    print(f'Successfully pushed to outputs/kaggle_runs/run_{timestamp}')
except Exception as e:
    print(f'GitHub push skipped or failed: {e}')
