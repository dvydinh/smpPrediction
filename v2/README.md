# V2 Architecture: Global Model (Single Model)

Trong thư mục này, kiến trúc 48 Mô hình độc lập (Direct Multi-Step) đã được chuyển đổi thành kiến trúc **1 Mô hình duy nhất (Global Model)**.

## Những thay đổi chính:
1. **Dữ liệu Dọc (Vertical Data):** Ma trận `Y` (N ngày x 48 chu kỳ) được duỗi dọc thành `(N * 48) x 1`. Tương tự cho tập đặc trưng `X`.
2. **Đặc trưng mới:** Bổ sung `target_cycle_id` (0 đến 47) và các biến thời gian lượng giác (`target_sin_hour`, `target_cos_hour`) để mô hình phân biệt được các chu kỳ trong ngày.
3. **Mô hình LightGBM Khổng lồ:** Tăng `num_leaves` lên `255` để mô hình có đủ không gian ghi nhớ quy luật của toàn bộ mốc thời gian. Số dòng huấn luyện giờ đây lên tới gần 100,000 dòng.
4. **Reshape Inference:** Sau khi dự báo mảng dọc, mảng kết quả sẽ được `reshape` trở lại thành `(N, 48)` để các khâu đánh giá MAPE, tính toán sai số, và vẽ biểu đồ phân tích tương đồng 100% với phiên bản cũ.

Tất cả các thông số đột phá từ bản gốc (như **Log-Residuals**, **Early Stopping 200**) đều được giữ nguyên.
