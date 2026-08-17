import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
from src.model_utils import StackingEnsemble

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stacking_ensemble.pkl"


def fetch_weather_forecast(target_date):
    print(f"Fetching Open-Meteo API for {target_date.strftime('%Y-%m-%d')}...")
    locations = {
        'hanoi': {'lat': 21.0285, 'lon': 105.8542},
        'danang': {'lat': 16.0678, 'lon': 108.2208},
        'hcmc': {'lat': 10.8231, 'lon': 106.6297}
    }

    weather_dict = {}
    for city, coords in locations.items():
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={coords['lat']}&longitude={coords['lon']}"
            f"&hourly=temperature_2m,relative_humidity_2m,cloud_cover,"
            f"wind_speed_10m,shortwave_radiation,direct_radiation,diffuse_radiation"
            f"&timezone=Asia%2FBangkok"
            f"&start_date={target_date.strftime('%Y-%m-%d')}"
            f"&end_date={target_date.strftime('%Y-%m-%d')}"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        hourly_keys = list(data['hourly'].keys())
        var_names = [
            'temperature', 'humidity', 'cloud_cover', 'wind_speed',
            'shortwave_radiation', 'direct_radiation', 'diffuse_radiation'
        ]
        for var_idx, var_name in enumerate(var_names):
            hourly_data = data['hourly'][hourly_keys[var_idx + 1]]
            half_hourly = np.repeat(hourly_data, 2)
            weather_dict[f'{var_name}_{city}'] = half_hourly

    return weather_dict


def predict_day_ahead(target_date_str=None):
    if target_date_str is None:
        current_date = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
    else:
        current_date = datetime.strptime(target_date_str, "%Y-%m-%d").replace(hour=7, minute=30)

    target_date = current_date + timedelta(days=1)

    print("Day-ahead forecast initialization")
    print(f"Snapshot: {current_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")

    print("Loading Stacking Ensemble model...")
    try:
        model = StackingEnsemble.load(str(MODEL_PATH))
    except Exception as e:
        print(f"[ERROR] Model not found at {MODEL_PATH}. Train on Kaggle first. Error: {e}")
        return

    # Load full historical data through the same pipeline used in training
    print("Loading historical data through preprocessing pipeline...")
    from src.data_preprocessing import get_data_paths, load_and_preprocess_data
    from src.feature_engineering import add_engineered_features

    DATA_ROOT, _ = get_data_paths()
    df = load_and_preprocess_data(DATA_ROOT)

    latest_observation = df['smp_system_price'].dropna().index.max()
    if latest_observation < current_date:
        raise RuntimeError(
            f"Historical market data ends at {latest_observation}. "
            f"Refresh raw inputs through {current_date} before forecasting."
        )

    df = df[df.index <= current_date].copy()

    # Fetch weather forecast for target day
    weather_forecast = fetch_weather_forecast(target_date)

    future_dates = [target_date + timedelta(minutes=30 * i) for i in range(48)]
    future_df = pd.DataFrame(index=pd.DatetimeIndex(future_dates, name='datetime'))

    # Fill weather columns from API
    for col, values in weather_forecast.items():
        future_df[col] = values[:48]

    future_df['smp_system_price'] = np.nan

    daily_cols = [c for c in df.columns if c.startswith('disp_')]
    for col in daily_cols:
        if col in df.columns:
            future_df[col] = df[col].iloc[-1]

    calendar_path = DATA_ROOT / 'exogenous' / 'calendar_vietnam.csv'
    calendar = pd.read_csv(calendar_path)
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar_row = calendar[calendar['date'] == pd.Timestamp(target_date.date())]
    calendar_cols = [
        'is_weekend', 'is_workday', 'is_holiday', 'is_tet',
        'is_pre_holiday', 'is_post_holiday', 'season'
    ]
    for col in calendar_cols:
        if col in calendar_row.columns and len(calendar_row):
            future_df[col] = calendar_row.iloc[0][col]

    gap_index = pd.date_range(
        current_date + timedelta(minutes=30),
        target_date - timedelta(minutes=30),
        freq='30min',
        name='datetime',
    )
    gap_df = pd.DataFrame(index=gap_index)

    full_df = pd.concat([df, gap_df, future_df])
    full_df = add_engineered_features(full_df)

    # Extract the target day (last 48 rows)
    X_48 = full_df.iloc[-48:].copy()
    feature_names = model.selected_features

    absent = [col for col in feature_names if col not in X_48.columns]
    if absent:
        raise RuntimeError(f"Absent production features: {absent}")

    X_48 = X_48[feature_names].replace([np.inf, -np.inf], np.nan).astype(float)
    missing = X_48.columns[X_48.isna().any()].tolist()
    if missing:
        raise RuntimeError(f"Missing production features: {missing}")

    # Predict
    print("Running inference...")
    y_pred = model.predict(X_48)

    print("SMP Forecast (VND/kWh)")
    print("-" * 35)
    print("Cycle | Time  | Price")
    print("-" * 35)
    for cycle in range(48):
        hour = cycle // 2
        minute = "30" if cycle % 2 != 0 else "00"
        time_str = f"{hour:02d}:{minute}"
        price = np.clip(y_pred[cycle], 0.0, 1778.6)
        print(f" {cycle:02d}    | {time_str} | {price:,.1f} VND")

    print("-" * 35)
    print("Inference complete. Ready for dispatch.")


if __name__ == "__main__":
    predict_day_ahead()
