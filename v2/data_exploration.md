# Tài liệu khai phá dữ liệu - V2 Global Model

## 1. Nguồn dữ liệu

Dataset VCGM (Vietnam Competitive Generation Market) được lưu trữ trên Kaggle tại `/kaggle/input/datasets/locccccc/vcgm-dataset`. Bao gồm 4 nhóm:

### 1.1. Dữ liệu thị trường (`market/`)
- **smp_prices_nsmo.csv**: giá SMP hệ thống, miền Bắc, miền Trung, miền Nam. Tần suất 30 phút.
- **dispatch_capacity_nsmo.csv**: công suất phát điện theo nguồn (thủy điện, điện mặt trời, điện gió, tổng). Tần suất ngày.

### 1.2. Dữ liệu thủy văn (`hydro/`)
- 66 file CSV (`hydro_hourly_*.csv`): lưu lượng nước đến, xả tổng, xả phát điện, xả tràn, mực nước hồ. Tần suất giờ.
- Được gộp (aggregate) theo `datetime` bằng `sum` (lưu lượng) và `mean` (mực nước), sau đó resample xuống 30 phút bằng forward-fill.

### 1.3. Dữ liệu thời tiết (`exogenous/weather/`)
- 3 thành phố: Hà Nội, Đà Nẵng, TP.HCM.
- 7 biến mỗi thành phố: nhiệt độ, độ ẩm, mây che phủ, tốc độ gió, bức xạ sóng ngắn, bức xạ trực tiếp, bức xạ khuếch tán.
- Tổng cộng: **21 biến thời tiết**.

> **⚠️ Giới hạn quan trọng:** Dữ liệu thời tiết trong dataset là **actual weather** (thời tiết thực tế đã xảy ra), không phải **forecast weather** (dự báo khí tượng). Snapshot lấy tại 07:30 ngày D nên biến thời tiết phản ánh điều kiện sáng hôm nay, không phải điều kiện ngày mai (D+1 — ngày cần dự báo). Trong môi trường production thực tế, phải thay thế bằng nguồn dữ liệu dự báo thời tiết D+1 từ API khí tượng.
>
> Lý do chưa sửa: dataset VCGM không cung cấp dữ liệu forecast. Mô hình vẫn hoạt động vì thời tiết có tính tự tương quan cao (autocorrelation) — thời tiết sáng nay là proxy hợp lý cho ngày mai trong điều kiện bình thường. Tuy nhiên, khi có front lạnh đột ngột hoặc bão, proxy này sẽ sai lệch nghiêm trọng.

### 1.4. Dữ liệu nhiên liệu (`exogenous/fuel/`)
- 5 biến: giá than proxy, giá dầu Brent, giá gas proxy, tỷ giá USD/VND, chỉ số DXY.
- Tần suất ngày, forward-fill xuống 30 phút.

## 2. Cấu trúc dữ liệu thô

- **Khung thời gian**: 01/01/2021 → 19/06/2026 (5.5 năm)
- **Tần suất**: 30 phút (48 chu kỳ/ngày)
- **Tổng số dòng**: 95.808
- **Master index**: `pd.date_range('2021-01-01', '2026-06-19 23:30', freq='30min')`

Tất cả các bảng dữ liệu con được join vào master index theo cột `datetime`, đảm bảo không bị lệch thời gian.

## 3. Feature engineering

### 3.1. Xử lý missing values
Thứ tự ưu tiên (chỉ dùng phương pháp nhìn về quá khứ, không nội suy tương lai):
1. Lấp bằng giá trị cùng giờ tuần trước (`shift(336)`)
2. Forward-fill (`ffill`) — kéo giá trị gần nhất từ quá khứ
3. Backward-fill (`bfill`) — chỉ cho vài dòng đầu tiên của dataset (01/2021) khi không có quá khứ

> **Lưu ý:** Phiên bản trước sử dụng `interpolate(method='linear')` — phương pháp này dùng cả điểm tương lai (t+1) để điền cho điểm hiện tại (t), gây ra data leakage tinh vi. Đã loại bỏ hoàn toàn.

### 3.2. Biến thời gian (cyclical encoding)
Sử dụng sin/cos để mã hóa tính tuần hoàn:
- `sin_hour`, `cos_hour` — chu kỳ trong ngày (24h)
- `sin_dow`, `cos_dow` — ngày trong tuần (7 ngày)
- `sin_month`, `cos_month` — tháng trong năm (12 tháng)

### 3.3. Biến trễ (lag features)
Tất cả lag được tính **trước** khi trích xuất snapshot, trên DataFrame gốc 30 phút:
- `smp_lag_48` (1 ngày trước), `smp_lag_49`, `smp_lag_50`
- `smp_lag_96` (2 ngày trước)
- `smp_lag_336` (7 ngày trước — cùng thứ tuần trước)
- Tương tự cho `load_lag_*` và `smp_north/south_lag_*`

### 3.3b. Dense lag vector (16 giá trị SMP gần nhất)
Để mô hình "nhìn" được hình dáng đường cong giá và ramp rate trước thời điểm snapshot:
- `smp_recent_1` đến `smp_recent_16`: 16 giá trị SMP liên tiếp từ 07:00 ngược về 23:30 đêm trước.
- Cho phép mô hình phát hiện xu hướng tăng/giảm, mức sàn/trần đêm qua, và tốc độ biến động.
- Tất cả đều nằm **trong** ngày D (trước snapshot 07:30), không leakage.

**Lưu ý quan trọng về LAG_SHIFT = 48:**
Tất cả rolling statistics (`smp_rolling_mean_24h`, `load_rolling_std_72h`,...) đều dùng `shift(48)` trước khi tính rolling. Điều này đảm bảo chỉ sử dụng dữ liệu từ **ít nhất 1 ngày trước**, tránh data leakage.

### 3.4. Biến vật lý (physics-informed)
- `solar_gen_proxy` = bức xạ × công suất lắp đặt × 0.75
- `wind_gen_proxy` = tốc độ gió × công suất lắp đặt × 0.3
- `residual_load_proxy` = tải tổng - solar_gen - wind_gen (dùng shift 48)
- `thermal_margin_proxy` = công suất nhiệt điện - residual_load
- `hydro_depletion_index` = mực nước hồ (shift 48)
- `margin_temp_interaction` = thermal_margin × nhiệt độ Đà Nẵng

### 3.5. Biến phân loại xác suất
- `is_spike` = 1 nếu SMP >= 1500 VND
- `is_zero` = 1 nếu SMP <= 100 VND
- `spike_prob_24h`, `spike_prob_72h` = tỷ lệ spike trong 24h/72h qua (shift 48)
- `zero_prob_24h`, `zero_prob_72h` = tương tự cho zero

### 3.6. Biến lịch
- `is_weekend`, `is_workday`, `is_holiday`, `is_tet`, `is_pre_holiday`, `is_post_holiday`
- `season` (1-4)

**Tổng số features sau engineering: 119 cột** (trước khi thêm cycle features). Bao gồm 16 dense lag mới.

## 4. Cách trích xuất X và Y

Mỗi mẫu (sample) trong mô hình tương ứng với **1 ngày dự báo**:
- **X (features)**: snapshot tại thời điểm **07:30 ngày D** — tức là thông tin mới nhất có thể biết được trước khi thị trường ngày D+1 mở cửa.
- **Y (target)**: 48 giá trị SMP của **ngày D+1** (từ 00:00 đến 23:30).

```
Snapshot: D 07:30   →   Predict: D+1 00:00, 00:30, ..., 23:30
```

Sau bước này: `X_daily.shape = (1995, 101)`, `Y_daily.shape = (1995, 48)`.

## 5. Residual learning (log-residuals với median baseline)

Thay vì dự đoán giá SMP trực tiếp, mô hình học **phần chênh lệch logarithmic** so với baseline:

```python
# Baseline = median của cùng thứ trong 3 tuần gần nhất (D-7, D-14, D-21)
for i in range(21, len(Y_daily)):
    candidates = [Y_daily[i-7], Y_daily[i-14], Y_daily[i-21]]
    Y_base[i] = np.median(candidates, axis=0)

Y_res = log1p(Y_daily) - log1p(Y_base)
```

Khi inference:
```
Y_pred = expm1(log1p(Y_base_test) + Y_pred_res)
```

> **Tại sao dùng median thay vì chỉ D-7?** Nếu đúng giờ đó tuần trước có sự cố nhà máy (trip), giá vọt lên 1778 VND. Dùng 1 ngày D-7 duy nhất sẽ khiến baseline bị nhiễu nặng, mô hình phải gánh residual cực âm. Median của 3 tuần khử nhiễu này hiệu quả.

Sau bước này: dataset bị cắt 21 dòng đầu → `X_daily.shape = (1974, 119)`.

## 6. Vertical flattening (kiến trúc V2)

Thay vì duy trì ma trận 2D `(N_days, 48)` và train 48 mô hình riêng, V2 duỗi dọc:

```python
X_daily = np.repeat(X_daily, 48, axis=0)     # (1988*48, 101)
Y_res   = Y_res.flatten()                     # (1988*48,)
```

Thêm 3 features mới cho mỗi dòng:
- `target_cycle_id` (0-47): chu kỳ nào trong ngày
- `target_sin_hour`, `target_cos_hour`: mã hóa lượng giác

**Sau flattening: `X.shape = (N*48, 122)`, `Y.shape = (N*48,)`** (con số chính xác phụ thuộc vào số ngày sau khi cắt 21 dòng đầu)

### Hạn chế đã biết
Mỗi ngày chỉ có 1 snapshot (07:30), nên cả 48 dòng sau khi flatten đều chia sẻ cùng giá trị lag. Mô hình dựa vào `target_cycle_id` để phân biệt hành vi giá giữa các giờ khác nhau. Đây không phải data leakage, nhưng là giới hạn thiết kế — mô hình không có lag riêng cho từng chu kỳ.

## 7. Phân tách train / validation / test

### 7.1. Cắt theo thời gian (temporal split)

```
dates_pd = pd.to_datetime(dates_arr)     # dates_arr đã được repeat 48 lần
train_mask = dates_pd <= '2026-03-31'
test_mask  = dates_pd >= '2026-04-01'
```

| Tập      | Khoảng thời gian         | Số ngày | Số dòng (sau flatten) |
|----------|--------------------------|---------|----------------------|
| Train    | 01/2021 → 03/2026        | ~1716   | 82.426               |
| Test     | 04/2026 → 06/2026        | 80      | 3.840                |

**Không có overlap**: train dùng `<=` 2026-03-31, test dùng `>=` 2026-04-01. Ranh giới rạch ròi.

### 7.2. Validation set (trích từ đuôi train)

```python
n_val = max(30, int(len(X_train) * 0.1))    # ~9158 dòng = ~190 ngày
X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]
X_vl, Y_vl = X_train[-n_val:], Y_train[-n_val:]
```

Validation nằm ở **cuối** tập train (gần nhất với test set), đảm bảo tính thời gian (temporal ordering). Không dùng random split.

### 7.3. Walk-forward cross validation

3 folds thời gian, mỗi fold train trên quá khứ, validate trên tương lai:

| Fold    | Train end  | Val start  | Val end    |
|---------|------------|------------|------------|
| Fold 1  | 2024-06-30 | 2024-07-01 | 2025-03-31 |
| Fold 2  | 2025-06-30 | 2025-07-01 | 2026-03-31 |
| Final   | 2026-03-31 | 2026-04-01 | 2026-06-19 |

Không có fold nào mà validation period nằm trước training period.

## 8. Kiểm tra data leakage

### 8.1. Lag features ✅ An toàn
- Tất cả lag ≥ 48 (tức ≥ 1 ngày). Snapshot lấy tại 07:30 ngày D, target là ngày D+1. Khoảng cách tối thiểu = 16.5 giờ.
- Rolling statistics dùng `shift(48)` trước khi tính → không nhìn thấy dữ liệu ngày hiện tại.

### 8.2. Train/test split ✅ An toàn
- Cắt thuần theo thời gian, không random shuffle.
- `train_mask` và `test_mask` dùng `<=` vs `>=` trên cùng một cột ngày, không có vùng chồng lấn.

### 8.3. Residual baseline (Y_base) ✅ An toàn
- `Y_base = median(Y_daily[i-7], Y_daily[i-14], Y_daily[i-21])` = trung vị giá cùng thứ trong 3 tuần gần nhất.
- Tất cả đều là dữ liệu quá khứ đã biết tại thời điểm dự báo.
- Median giúp khử nhiễu từ spike/trip đơn lẻ, ổn định hơn so với chỉ dùng D-7.

### 8.4. Vertical flattening ✅ An toàn
- `np.repeat` chỉ nhân bản dòng features cũ, không tạo ra thông tin mới.
- `target_cycle_id` là metadata (biết trước), không phải target leakage.

## 9. Thống kê mô tả tóm tắt

| Chỉ số                        | Giá trị        |
|--------------------------------|----------------|
| Phạm vi dữ liệu               | 01/2021 – 06/2026 |
| Tổng số ngày                   | 1995           |
| Tổng số dòng thô (30 phút)    | 95.808         |
| Số features gốc                | 119            |
| Số features sau flatten        | 122            |
| Số dòng train (flatten)        | ~81.700 (ước tính) |
| Số dòng test (flatten)         | 3.840          |
| Tỷ lệ test / tổng             | ~4%            |
| Giá SMP trung bình             | ~1.370 VND     |
| Giá SMP cap                    | 1.778,6 VND    |
| Ngưỡng spike                   | ≥ 1.500 VND    |
| Ngưỡng zero                    | ≤ 100 VND      |
