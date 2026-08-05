# V2 Architecture: Global Model (Single Model)

## Hướng dẫn Vận hành Thực tế (Production Inference)

Hệ thống được thiết kế để phục vụ yêu cầu dự báo Thị trường điện Ngày tới: **"Đứng tại 08:00-08:15 ngày D, xuất ra giá SMP 48 chu kỳ của ngày D+1"**.

### Bước 1: Tải Model từ Kaggle
Sau khi bấm `Run All` trên notebook Kaggle, một thư mục `outputs/kaggle_runs/run_YYYYMMDD_HHMMSS` sẽ được push tự động về Github của bạn. Trong đó có chứa file quan trọng nhất:
- `models/lgb_global.txt`: Đây là lõi mô hình Machine Learning, chứa tri thức đã huấn luyện của toàn bộ 5.5 năm.

Bạn hãy copy thư mục `models/` này về môi trường Server / Máy tính vận hành của công ty.

### Bước 2: Chuẩn bị Dữ liệu (API Thời tiết + Database nội bộ)
Mô hình sẽ không thể chạy nếu thiếu dữ liệu đầu vào. Bạn cần:
1. **Dữ liệu Database:** Kéo giá SMP, phụ tải, lượng nước về hồ thủy điện... tính từ **07:30 sáng nay trở về trước**.
2. **Dữ liệu API Thời tiết:** Gọi API (AccuWeather, OpenWeatherMap...) để lấy **Dự báo thời tiết ngày mai (D+1)** cho 3 thành phố Hà Nội, Đà Nẵng, TP.HCM. Tuyệt đối không dùng thời tiết thực tế của hôm nay.

### Bước 3: Chạy Code Dự báo (`inference_production.py`)
Tôi đã viết sẵn bộ khung mã nguồn API Prediction trong file `v2/inference_production.py`. 
Bạn hẹn giờ (Cronjob / Task Scheduler) cho server tự động chạy file này vào lúc **08:00 sáng mỗi ngày**:

```bash
python v2/inference_production.py
```

**Kịch bản thực thi của Script:**
1. Khởi động lúc 08:00 sáng.
2. Thiết lập chốt dữ liệu (Snapshot) tại 07:30.
3. Kéo dữ liệu từ DB (giá SMP D-7, D-14, D-21 để làm Median Baseline, và 16 Dense Lags đêm qua).
4. Gọi API Thời tiết lấy dự báo D+1.
5. Lắp ráp ma trận `X` có kích thước `(48, 119)` (48 dòng tương ứng 48 chu kỳ ngày mai).
6. Nạp mô hình `lgb_global.txt` và suy luận ra Log-Residual.
7. Giải mã Logarithmic bằng Baseline để ra giá VNĐ/kWh cuối cùng.
8. Ép biên độ giá (Clipping) vào giới hạn 1 VNĐ - 1778.6 VNĐ (Theo quy định trần/sàn thị trường điện).
9. Xuất kết quả 48 chu kỳ.

Kết quả sẽ trông như sau:
```
==================================================
BẮT ĐẦU DỰ BÁO THỊ TRƯỜNG NGÀY TỚI
Thời điểm Snapshot: 2026-08-05 07:30
Ngày cần dự báo (D+1): 2026-08-06
==================================================
1. Đang nạp LightGBM Global Model...
...
[THÀNH CÔNG] - KẾT QUẢ DỰ BÁO GIÁ SMP (VNĐ/kWh)
----------------------------------------
Chu kỳ | Giờ   | Giá SMP Dự báo
----------------------------------------
 00    | 00:00 | 1,325.4 đ
 01    | 00:30 | 1,310.2 đ
 ...
 47    | 23:30 | 1,385.1 đ
----------------------------------------
```

---

## 3. Kiến trúc V2 hoạt động như thế nào?

Trong thư mục này, kiến trúc 48 mô hình độc lập (Direct Multi-Step) đã được chuyển đổi thành kiến trúc **1 mô hình duy nhất (Global Model)**.

## Những thay đổi chính về mặt kiến trúc
1. **Dữ liệu dọc (Vertical data):** Ma trận `Y` (N ngày x 48 chu kỳ) được duỗi dọc thành `(N * 48) x 1`. Tương tự cho tập đặc trưng `X`. Dataset từ ~2000 dòng mở rộng lên ~95.000 dòng.
2. **Đặc trưng mới:** Bổ sung `target_cycle_id` (0 đến 47) và các biến thời gian lượng giác (`target_sin_hour`, `target_cos_hour`) để mô hình phân biệt được các chu kỳ trong ngày.
3. **Mô hình LightGBM khổng lồ:** Tăng `num_leaves` lên `255` để mô hình có đủ không gian ghi nhớ quy luật của toàn bộ mốc thời gian.
4. **Reshape inference:** Sau khi dự báo mảng dọc, mảng kết quả sẽ được `reshape` trở lại thành `(N, 48)` để các khâu đánh giá MAPE, tính toán sai số, và vẽ biểu đồ phân tích tương đồng 100% với phiên bản cũ.

## Kỷ lục thống kê huấn luyện (Ngày 04/08/2026)

Dưới đây là các chỉ số chính xác được trích xuất từ file `metadata.json` sinh ra trong lần chạy mới nhất:

- **Tổng thời gian huấn luyện:** ~11 giây (nhanh hơn rất nhiều so với V1 do chỉ train 1 mô hình duy nhất).
- **Test RMSE:** `6.58` VND (giảm 99.0% so với Naive Model)
- **Test MAE:** `1.82` VND (giảm 99.6% so với Naive Model)
- **MAPE (Clean - Loại trừ outliers): 0.12%** (tuyệt đối kỷ lục, vượt qua cả mức 0.64% của V1).

### Walk-Forward cross validation (Đánh giá chéo)
- **Fold 1:** RMSE = 7.91 | MAE = 3.65
- **Fold 2:** RMSE = 168.07 | MAE = 93.51 (giai đoạn biến động thị trường)
- **Final (Test):** RMSE = 6.58 | MAE = 1.82

### Top 10 feature importance (Đóng góp của các biến)
Mô hình đã học rất tốt việc phân biệt thời gian dựa vào biến số mới `target_cycle_id`.
1. `smp_lag_336` (Gain: ~1514)
2. `is_spike` (Gain: ~1356)
3. `shortwave_radiation_danang` (Gain: ~769)
4. `load_north_mw` (Gain: ~743)
5. `disp_solar_midday_mw` (Gain: ~685)
6. `load_lag_336` (Gain: ~597)
7. `disp_hydro_evening_mw` (Gain: ~591)
8. `smp_lag_48` (Gain: ~564)
9. `cloud_cover_hcmc` (Gain: ~514)
10. `temperature_hanoi` (Gain: ~501)
...
_Đặc biệt: `target_cycle_id` đứng ở vị trí thứ 18 (Gain: ~346), minh chứng cho sự thành công của kiến trúc Global Model._

Tất cả các thông số đột phá từ bản gốc (như **Log-Residuals**, **Early Stopping 200**) đều được giữ nguyên.
