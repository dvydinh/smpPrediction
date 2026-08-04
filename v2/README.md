# V2 Architecture: Global Model (Single Model)

Trong thư mục này, kiến trúc 48 Mô hình độc lập (Direct Multi-Step) đã được chuyển đổi thành kiến trúc **1 Mô hình duy nhất (Global Model)**.

## Những thay đổi chính về mặt Kiến trúc
1. **Dữ liệu Dọc (Vertical Data):** Ma trận `Y` (N ngày x 48 chu kỳ) được duỗi dọc thành `(N * 48) x 1`. Tương tự cho tập đặc trưng `X`. Dataset từ ~2000 dòng mở rộng lên ~95.000 dòng.
2. **Đặc trưng mới:** Bổ sung `target_cycle_id` (0 đến 47) và các biến thời gian lượng giác (`target_sin_hour`, `target_cos_hour`) để mô hình phân biệt được các chu kỳ trong ngày.
3. **Mô hình LightGBM Khổng lồ:** Tăng `num_leaves` lên `255` để mô hình có đủ không gian ghi nhớ quy luật của toàn bộ mốc thời gian.
4. **Reshape Inference:** Sau khi dự báo mảng dọc, mảng kết quả sẽ được `reshape` trở lại thành `(N, 48)` để các khâu đánh giá MAPE, tính toán sai số, và vẽ biểu đồ phân tích tương đồng 100% với phiên bản cũ.

## Kỷ lục Thống kê Huấn luyện (Ngày 04/08/2026)

Dưới đây là các chỉ số chính xác được trích xuất từ file `metadata.json` sinh ra trong lần chạy mới nhất:

- **Tổng thời gian Huấn luyện:** ~11 giây (Nhanh hơn rất nhiều so với V1 do chỉ train 1 mô hình duy nhất).
- **Test RMSE:** `6.58` VND (Giảm 99.0% so với Naive Model)
- **Test MAE:** `1.82` VND (Giảm 99.6% so với Naive Model)
- **MAPE (Clean - Loại trừ Outliers): 0.12%** (Tuyệt đối kỷ lục, vượt qua cả mức 0.64% của V1).

### Walk-Forward Cross Validation (Đánh giá chéo)
- **Fold 1:** RMSE = 7.91 | MAE = 3.65
- **Fold 2:** RMSE = 168.07 | MAE = 93.51 (Giai đoạn biến động thị trường)
- **Final (Test):** RMSE = 6.58 | MAE = 1.82

### Top 10 Feature Importance (Đóng góp của các Biến)
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
