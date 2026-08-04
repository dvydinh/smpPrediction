with open('doc.md', 'a', encoding='utf-8') as f:
    f.write('''
## 10. Ghi nhận Kết quả Vòng 4 - Tối ưu Toán học Log-Residuals (04/08/2026)
- **Vấn đề trước Vòng 4:** Lỗi loại bỏ Outlier đã phơi bày sai số ở tệp giá bình thường, phát hiện MAPE thực tế là 13.80%. Tuy thấp so với mặt bằng chung nhưng chưa đạt mốc dưới 10%.
- **Chiến thuật triển khai:**
  1. **Hạn chế Premature Early Stopping:** Giảm `learning_rate` xuống `0.005` và tăng `early_stopping` lên `200` để ép các Cây quyết định phải đào sâu hơn, cấm mô hình bỏ cuộc sớm ở Cây số 1 vào ban đêm.
  2. **Vũ khí Log-Residuals (Triệt tiêu MAPE):** Thay đổi mục tiêu học thuật từ độ lệch tuyệt đối sang hàm logarit phần dư: $Y_{res} = \\log(Y_{actual} + 1) - \\log(Y_{base} + 1)$. Khi đó quá trình giải mã sử dụng `np.expm1`. Lý thuyết toán học đằng sau kỹ thuật này là: Việc giảm thiểu sai lệch tuyệt đối (MAE) trên không gian Logarit sẽ trực tiếp tạo ra sai số phần trăm (MAPE) nhỏ nhất trên không gian thực. Kỹ thuật này được áp dụng như vũ khí tối thượng cho các bài toán yêu cầu giới hạn %.
  3. **Fair Evaluation Block:** Nhúng hệ thống tính toán (đã loại bỏ Outlier 100-1778 VND) bằng tiếng Anh vào Kaggle để đánh giá tự động trong mỗi lần huấn luyện, đảm bảo tính công bằng và chuyên nghiệp.
''')
