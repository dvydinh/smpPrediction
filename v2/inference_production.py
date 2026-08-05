"""
Production Inference Script cho SMP Prediction (V2 Global Model)
Yêu cầu: Chạy hàng ngày vào lúc 08:00 sáng ngày D.
Mục tiêu: Dự báo giá SMP cho 48 chu kỳ của ngày D+1 (từ 00:00 đến 23:30).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# Cấu hình đường dẫn
MODEL_PATH = "models/lgb_global.txt"

def fetch_weather_forecast(target_date):
    """
    Giả lập gọi API Thời tiết (Ví dụ: OpenWeatherMap, AccuWeather).
    Trong thực tế, bạn thay bằng HTTP GET request thực.
    """
    print(f"[API] Đang lấy dự báo thời tiết cho ngày {target_date.strftime('%Y-%m-%d')}...")
    # Mock data: 21 biến thời tiết (7 biến x 3 thành phố)
    # Nhiệt độ, độ ẩm, mây che phủ, tốc độ gió, bức xạ sóng ngắn, bức xạ trực tiếp, khuếch tán
    weather_features = {
        'temp_hanoi': 28.5, 'humidity_hanoi': 75.0, 'cloud_hanoi': 20.0, 'wind_hanoi': 3.5, 
        'shortwave_hanoi': 450.0, 'direct_hanoi': 300.0, 'diffuse_hanoi': 150.0,
        # ... (tương tự cho Đà Nẵng, TP.HCM)
    }
    # Điền giá trị giả lập cho đủ 21 biến (để script không lỗi khi minh họa)
    for city in ['danang', 'hcmc']:
        for var in ['temp', 'humidity', 'cloud', 'wind', 'shortwave', 'direct', 'diffuse']:
            weather_features[f'{var}_{city}'] = weather_features[f'{var}_hanoi']
            
    return weather_features

def fetch_historical_data_from_db(current_date):
    """
    Kéo dữ liệu từ Database nội bộ của công ty (SQL/Data Warehouse).
    Bao gồm: giá SMP quá khứ, phụ tải, thủy văn tính đến 07:30 sáng nay.
    """
    print(f"[DB] Kéo dữ liệu lịch sử tính đến 07:30 ngày {current_date.strftime('%Y-%m-%d')}...")
    
    # 1. Để tính Median Baseline (Y_base), ta cần đúng giá SMP của D-7, D-14, D-21
    # Trả về 48 giá trị của mỗi ngày
    y_d_7 = np.full(48, 1450.0) 
    y_d_14 = np.full(48, 1420.0)
    y_d_21 = np.full(48, 1480.0)
    
    y_base = np.median([y_d_7, y_d_14, y_d_21], axis=0) # Shape: (48,)
    
    # 2. Để tính 16 Dense Lags (smp_recent_1 -> 16), ta cần giá SMP từ 07:00 lùi về 23:30 đêm qua
    smp_recent = np.full(16, 1300.0) # Mock 16 giá trị
    
    # 3. Các feature tĩnh khác (chỉ cần 1 dòng đại diện cho snapshot lúc 07:30)
    # Ví dụ: phụ tải trễ (load_lag_48), trạng thái hồ thủy điện, v.v.
    snapshot_features = {
        'smp_lag_48': 1350.0,
        'smp_lag_96': 1360.0,
        'smp_lag_336': 1450.0,
        'hydro_inflow_sum': 12000.0,
        'hydro_level_mean': 150.5,
        'is_weekend': 0 if current_date.weekday() < 5 else 1,
        # ... (thêm đủ 119 features tĩnh ngoại trừ các feature chu kỳ)
    }
    
    return y_base, smp_recent, snapshot_features

def predict_day_ahead(target_date_str=None):
    """
    Hàm thực thi chính để chạy dự báo lúc 08:00 sáng.
    """
    if target_date_str is None:
        # Chạy tự động thì lấy ngày hiện tại (Day D)
        current_date = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
    else:
        # Chạy manual cho ngày cụ thể
        current_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        
    target_date = current_date + timedelta(days=1) # D+1
    
    print("="*50)
    print(f"BẮT ĐẦU DỰ BÁO THỊ TRƯỜNG NGÀY TỚI")
    print(f"Thời điểm Snapshot: {current_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Ngày cần dự báo (D+1): {target_date.strftime('%Y-%m-%d')}")
    print("="*50)
    
    # 1. Nạp Model
    print("1. Đang nạp LightGBM Global Model...")
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
    except Exception as e:
        print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}. Vui lòng train mô hình trên Kaggle và tải file lgb_global.txt về.")
        return
        
    # 2. Lấy dữ liệu API và Database
    weather_forecast = fetch_weather_forecast(target_date)
    y_base, smp_recent, snapshot_features = fetch_historical_data_from_db(current_date)
    
    # 3. Lắp ráp ma trận X cho 48 chu kỳ
    print("3. Đang nội suy và lắp ráp ma trận 48 chu kỳ...")
    # Khởi tạo ma trận (48 dòng, 122 cột - giả định số lượng features)
    # Trong thực tế, bạn cần sắp xếp đúng thứ tự cột như file model.feature_name()
    feature_names = model.feature_name()
    X_48 = pd.DataFrame(index=range(48), columns=feature_names)
    X_48 = X_48.fillna(0.0) # Mock điền 0
    
    # Gắn các thông số chung (weather, lags, snapshot) vào toàn bộ 48 chu kỳ
    for col in feature_names:
        if col in weather_forecast:
            X_48[col] = weather_forecast[col]
        elif col in snapshot_features:
            X_48[col] = snapshot_features[col]
            
    for i in range(16):
        col_name = f'smp_recent_{i+1}'
        if col_name in feature_names:
            X_48[col_name] = smp_recent[i]
    
    # Gắn các thông số chu kỳ (Dynamic cycle features)
    X_48['target_cycle_id'] = range(48)
    X_48['target_sin_hour'] = np.sin(2 * np.pi * X_48['target_cycle_id'] / 48.0)
    X_48['target_cos_hour'] = np.cos(2 * np.pi * X_48['target_cycle_id'] / 48.0)
    
    # 4. Dự báo Residual
    print("4. Đang chạy mô hình suy luận (Inference)...")
    y_pred_res = model.predict(X_48)
    
    # 5. Khôi phục giá trị thực từ Log-Residual và Y_base
    # Công thức: Y_pred = expm1( log1p(Y_base) + Y_pred_res )
    y_pred_final = np.expm1(np.log1p(y_base) + y_pred_res)
    
    # 6. Hiển thị kết quả
    print("\n[THÀNH CÔNG] - KẾT QUẢ DỰ BÁO GIÁ SMP (VNĐ/kWh)")
    print("-" * 40)
    print("Chu kỳ | Giờ   | Giá SMP Dự báo")
    print("-" * 40)
    for cycle in range(48):
        hour = cycle // 2
        minute = "30" if cycle % 2 != 0 else "00"
        time_str = f"{hour:02d}:{minute}"
        
        # Clip giá dự báo trong giới hạn trần/sàn của thị trường điện VN
        price = np.clip(y_pred_final[cycle], 1.0, 1778.6)
        
        print(f" {cycle:02d}    | {time_str} | {price:,.1f} đ")
        
    print("-" * 40)
    print("Vui lòng xuất kết quả ra file Excel / gửi Email tới phòng Vận hành thị trường điện.")

if __name__ == "__main__":
    # Đặt script này vào cron job chạy lúc 08:00 AM mỗi ngày
    predict_day_ahead()
