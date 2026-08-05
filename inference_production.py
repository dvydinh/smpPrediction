import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import warnings
from datetime import datetime, timedelta

MODEL_PATH = "models/lgb_global.txt"

def fetch_weather_forecast(target_date):
    print(f"[INFO] Fetching_Open_Meteo_forecast: {target_date.strftime('%Y-%m-%d')}...")
    locations = {
        'hanoi': {'lat': 21.0285, 'lon': 105.8542},
        'danang': {'lat': 16.0678, 'lon': 108.2208},
        'hcmc': {'lat': 10.7626, 'lon': 106.6602}
    }
    
    variables = "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation,shortwave_radiation,direct_radiation,diffuse_radiation"
    
    weather_dict = {}
    for city, coords in locations.items():
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&hourly={variables}&timezone=Asia%2FBangkok"
        response = requests.get(url).json()
        
        df_weather = pd.DataFrame(response['hourly'])
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df_target = df_weather[df_weather['time'].dt.date == target_date.date()].copy()
        
        weather_dict[f'temperature_{city}'] = np.repeat(df_target['temperature_2m'].values, 2)
        weather_dict[f'humidity_{city}'] = np.repeat(df_target['relative_humidity_2m'].values, 2)
        weather_dict[f'wind_speed_{city}'] = np.repeat(df_target['wind_speed_10m'].values, 2)
        weather_dict[f'cloud_cover_{city}'] = np.repeat(df_target['cloud_cover'].values, 2)
        weather_dict[f'precipitation_{city}'] = np.repeat(df_target['precipitation'].values, 2)
        weather_dict[f'shortwave_radiation_{city}'] = np.repeat(df_target['shortwave_radiation'].values, 2)
        
        if city == 'hanoi' or city == 'hcmc':
            weather_dict[f'direct_radiation_{city}'] = np.repeat(df_target['direct_radiation'].values, 2)
            weather_dict[f'diffuse_radiation_{city}'] = np.repeat(df_target['diffuse_radiation'].values, 2)
            
    return weather_dict

def fetch_historical_data_from_db(current_date):
    print(f"[INFO] Fetching_historical_DB: {current_date.strftime('%Y-%m-%d')}...")
    
    smp_path = "data/raw/market/smp_prices_nsmo.csv"
    try:
        smp = pd.read_csv(smp_path)
        smp['datetime'] = pd.to_datetime(smp['datetime'])
        smp = smp.sort_values('datetime').set_index('datetime')
    except Exception as e:
        print(f"[WARN] Historical_data_not_found. Using_mock_data. {smp_path}. Đang dùng Mock Data.")
        smp = pd.DataFrame()
        
    y_d_7 = np.full(48, np.nan)
    y_d_14 = np.full(48, np.nan)
    y_d_21 = np.full(48, np.nan)
    
    if not smp.empty:
        try:
            target_date = current_date + timedelta(days=1)
            for d, arr in zip([7, 14, 21], [y_d_7, y_d_14, y_d_21]):
                past_date_start = (target_date - timedelta(days=d)).replace(hour=0, minute=0)
                past_date_end = past_date_start + timedelta(hours=23, minutes=30)
                subset = smp.loc[past_date_start:past_date_end, 'smp_system_price']
                if len(subset) == 48:
                    arr[:] = subset.values
        except Exception:
            pass
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_base = np.nanmedian([y_d_7, y_d_14, y_d_21], axis=0)
    y_base = np.nan_to_num(y_base, nan=1300.0)
    
    smp_recent = np.full(16, np.nan)
    if not smp.empty:
        try:
            recent_subset = smp.loc[:current_date, 'smp_system_price'].tail(16)
            if len(recent_subset) == 16:
                smp_recent[:] = recent_subset.values[::-1] 
        except Exception:
            pass
    
    snapshot_features = {
        'is_weekend': 0 if current_date.weekday() < 5 else 1,
    }
    
    return y_base, smp_recent, snapshot_features

def predict_day_ahead(target_date_str=None):
    if target_date_str is None:
        current_date = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
    else:
        current_date = datetime.strptime(target_date_str, "%Y-%m-%d").replace(hour=7, minute=30)
        
    target_date = current_date + timedelta(days=1)
    
    print("="*50)
    print(f"DAY-AHEAD_MARKET_FORECAST_INITIATED")
    print(f"[INFO] Snapshot_time: {current_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"[INFO] Target_forecast_date: {target_date.strftime('%Y-%m-%d')}")
    print("="*50)
    
    print("[STEP 1] Loading_LightGBM_Global_Model...")
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Model_not_found_at {MODEL_PATH}. Please_train_model_and_download_lgb_global.txt.")
        return
        
    weather_forecast = fetch_weather_forecast(target_date)
    y_base, smp_recent, snapshot_features = fetch_historical_data_from_db(current_date)
    
    print("[STEP 3] Assembling_48-cycle_matrix...")
    feature_names = model.feature_name()
    X_48 = pd.DataFrame(index=range(48), columns=feature_names)
    
    for col in feature_names:
        if col in weather_forecast:
            X_48[col] = weather_forecast[col]
        elif col in snapshot_features:
            X_48[col] = snapshot_features[col]
            
    for i in range(16):
        col_name = f'smp_recent_{i+1}'
        if col_name in feature_names:
            X_48[col_name] = smp_recent[i]
    
    X_48['target_cycle_id'] = range(48)
    X_48['target_sin_hour'] = np.sin(2 * np.pi * X_48['target_cycle_id'] / 48.0)
    X_48['target_cos_hour'] = np.cos(2 * np.pi * X_48['target_cycle_id'] / 48.0)
    
    print("[STEP 4] Running_inference...")
    X_48 = X_48.astype(float)
    y_pred_res = model.predict(X_48)
    
    y_pred_final = np.expm1(np.log1p(y_base) + y_pred_res)
    
    print("[SUCCESS] SMP_PRICE_FORECAST_RESULTS (VND/kWh)")
    print("-" * 40)
    print("Cycle | Time  | Predicted_SMP")
    print("-" * 40)
    for cycle in range(48):
        hour = cycle // 2
        minute = "30" if cycle % 2 != 0 else "00"
        time_str = f"{hour:02d}:{minute}"
        
        price = np.clip(y_pred_final[cycle], 1.0, 1778.6)
        
        print(f" {cycle:02d}    | {time_str} | {price:,.1f} đ")
        
    print("-" * 40)
    print("[INFO] Ready_for_export/email_dispatch.")

if __name__ == "__main__":
    predict_day_ahead()

