import os, glob, gc, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
DATA_ROOT = Path('data/raw')
MARKET_DIR = DATA_ROOT / 'market'
HYDRO_DIR  = DATA_ROOT / 'hydro'
EXOG_DIR   = DATA_ROOT / 'exogenous'
OUTPUT_DIR = Path('outputs')
PRICE_CAP = 1778.6
NEAR_ZERO = 2.0
CYCLES_PER_DAY = 48
LAG_SHIFT = 48
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
for col in df.select_dtypes(include=[np.number]).columns:
    if not df[col].isna().any(): continue
    df[col] = df[col].interpolate(method='linear', limit=2)
    if df[col].isna().any(): df[col] = df[col].fillna(df[col].shift(336))
    if df[col].isna().any(): df[col] = df[col].ffill().bfill()

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
df = df.copy()
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
# --- RESIDUAL LEARNING SETUP ---
X_daily = X_daily[7:]
Y_base  = Y_daily[:-7]
Y_daily = Y_daily[7:]
dates_arr = dates_arr[7:]
Y_res = Y_daily - Y_base
print(f'Residuals prepared. X: {X_daily.shape}')

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
models = {}
for k in range(CYCLES_PER_DAY):
    print(f'Loading model {k}...'); models[k] = lgb.Booster(model_file=f'outputs/kaggle_runs/run_20260804_142254/models/lgb_cycle_{k:02d}.txt')
print('Loaded 48 models locally.')
Y_pred_res = np.zeros_like(Y_test)
for k in range(CYCLES_PER_DAY):
    Y_pred_res[:, k] = models[k].predict(X_test)
Y_pred = np.clip(Y_pred_res + Y_base_test, 0, PRICE_CAP)
Y_test_actual = Y_test + Y_base_test

test_rmse = np.sqrt(mean_squared_error(Y_test_actual.flatten(), Y_pred.flatten()))
test_mae  = mean_absolute_error(Y_test_actual.flatten(), Y_pred.flatten())

print(f'Test RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}')
print(f'Naive RMSE: {naive_rmse:.2f} | MAE: {naive_mae:.2f}')
print(f'Improvement: {(1 - test_rmse/naive_rmse)*100:.1f}% RMSE, {(1 - test_mae/naive_mae)*100:.1f}% MAE')

cycle_rmses, cycle_maes = [], []
for k in range(CYCLES_PER_DAY):
    cycle_rmses.append(np.sqrt(mean_squared_error(Y_test_actual[:, k], Y_pred[:, k])))
    cycle_maes.append(mean_absolute_error(Y_test_actual[:, k], Y_pred[:, k]))
test_dates = dates_pd[test_mask]
n_sample = min(6, len(test_dates))
sample_idx = np.linspace(0, len(test_dates) - 1, n_sample, dtype=int)
tick_pos = range(0, 48, 4)
tick_labels = [f'{c//2:02d}:{(c%2)*30:02d}' for c in tick_pos]

fig, axes = plt.subplots(n_sample, 1, figsize=(16, 4 * n_sample))
if n_sample == 1: axes = [axes]
for ax, i in zip(axes, sample_idx):
    ax.plot(range(48), Y_test[i], 'b-o', ms=3, lw=1.5, label='Actual', alpha=0.8)
    ax.plot(range(48), Y_pred[i], 'r--s', ms=3, lw=1.5, label='Predicted', alpha=0.8)
    ax.set_title(test_dates[i].strftime('%Y-%m-%d (%A)'), fontweight='bold')
    ax.set(ylabel='SMP (VND)'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
residuals = Y_test.flatten() - Y_pred.flatten()
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

axes[0, 0].scatter(range(len(residuals)), residuals, s=1, alpha=0.3, c='steelblue')
axes[0, 0].axhline(0, c='red', lw=1); axes[0, 0].set_title('Residuals')

axes[0, 1].hist(residuals, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
axes[0, 1].axvline(0, c='red', lw=1); axes[0, 1].set_title('Distribution')

from statsmodels.tsa.stattools import acf
r_acf = acf(residuals, nlags=100)
ci = 1.96 / np.sqrt(len(residuals))
axes[1, 0].bar(range(len(r_acf)), r_acf, color='darkorange', alpha=0.7)
axes[1, 0].axhline(ci, ls='--', c='blue', alpha=0.5)
axes[1, 0].axhline(-ci, ls='--', c='blue', alpha=0.5); axes[1, 0].set_title('ACF')

axes[1, 1].bar(range(48), cycle_rmses, color='steelblue', alpha=0.7)
axes[1, 1].set_xticks(tick_pos); axes[1, 1].set_xticklabels(tick_labels, rotation=45)
axes[1, 1].set_title('RMSE by cycle')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Residuals: mean={residuals.mean():.2f}, std={residuals.std():.2f}')
gain = np.zeros(len(feature_cols))
for k in range(CYCLES_PER_DAY):
    gain += models[k].feature_importance(importance_type='gain')
gain /= CYCLES_PER_DAY

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

mask_outliers = (Y_test_actual > 100) & (Y_test_actual < 1778.0)
y_true_clean = Y_test_actual[mask_outliers]
y_pred_clean = Y_pred[mask_outliers]
mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
print('\n--- KẾT QUẢ ĐÃ LỌC OUTLIER ---')
print(f'Mẫu hợp lệ: {len(y_true_clean)} / {Y_test_actual.size}')
print(f'MAE (Clean): {mae_clean:.2f} VND')
print(f'MAPE (Clean): {mape:.2f}%')
