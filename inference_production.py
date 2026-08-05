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
        response = requests.get(url)
        if response.status_code == 200:
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

    # Load model
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

    # Truncate to the snapshot time (08:00 Day D means we have up to 07:30 Day D)
    df = df[df.index <= current_date].copy()

    # Fetch weather forecast for target day
    weather_forecast = fetch_weather_forecast(target_date)

    # Create 48 rows for Day D+1
    future_dates = [target_date + timedelta(minutes=30 * i) for i in range(48)]
    future_df = pd.DataFrame(index=pd.DatetimeIndex(future_dates, name='datetime'))

    # Fill weather columns from API
    for col, values in weather_forecast.items():
        future_df[col] = values[:48]

    future_df['smp_system_price'] = np.nan

    # Forward-fill dispatch and other daily columns from the last known day
    daily_cols = [c for c in df.columns if c.startswith('disp_') or c in [
        'coal_proxy_price', 'brent_price', 'gas_proxy_price', 'usd_vnd', 'dxy_index',
        'is_weekend', 'is_workday', 'is_holiday', 'is_tet',
        'is_pre_holiday', 'is_post_holiday', 'season'
    ]]
    for col in daily_cols:
        if col in df.columns:
            future_df[col] = df[col].iloc[-1]

    # Forward-fill load from the same cycle yesterday as a proxy
    if 'load_total_mw' in df.columns:
        last_48 = df['load_total_mw'].iloc[-48:].values
        future_df['load_total_mw'] = last_48

    for col in ['load_north_mw', 'load_central_mw', 'load_south_mw']:
        if col in df.columns:
            future_df[col] = df[col].iloc[-48:].values

    # Concat and engineer features
    full_df = pd.concat([df, future_df])
    full_df = add_engineered_features(full_df)

    # Extract the target day (last 48 rows)
    X_48 = full_df.iloc[-48:].copy()
    # The feature names can be retrieved from the LGBM base model
    feature_names = model.models['lgb'].feature_name_

    # Ensure all required features exist
    for col in feature_names:
        if col not in X_48.columns:
            X_48[col] = 0.0

    X_48 = X_48[feature_names].astype(float)

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
