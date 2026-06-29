# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

> Người thực hiện: **Bùi Đình Mạnh**

---

## 6.1. Kết luận

Đề tài **"Đánh giá nguy cơ mắc bệnh tim bằng Mạng nơ-ron nhân tạo (ANN)"** đã
được thực hiện hoàn chỉnh theo đúng quy trình của một dự án học máy:

1. **Dữ liệu:** thu thập bộ Heart Disease (Kaggle), tiền xử lý kỹ lưỡng — phát
   hiện và loại bỏ **723 bản ghi trùng lặp** (còn 302 bản duy nhất), xử lý ngoại
   lệ, chuẩn hoá đặc trưng và chia tập train/test 80/20.
2. **Mô hình:** thiết kế và huấn luyện Mạng nơ-ron nhân tạo nhiều tầng
   (13 → 64 → 32 → 16 → 1) với ReLU, Sigmoid, Adam, Binary Crossentropy và
   Early Stopping.
3. **Đánh giá:** mô hình đạt **Accuracy ≈ 0.74**, **F1-Score ≈ 0.76** và
   **ROC-AUC ≈ 0.85** trên tập test; thử nghiệm 4 cấu hình siêu tham số và chọn
   được mô hình tốt nhất (**ANN-3**, F1 ≈ 0.79, Recall ≈ 0.82).

**Kết quả cho thấy:** mạng nơ-ron nhân tạo là một hướng tiếp cận **khả thi và
hiệu quả** cho bài toán sàng lọc nguy cơ bệnh tim từ dữ liệu y tế dạng bảng, với
khả năng phân tách hai lớp tốt (ROC-AUC ≈ 0.85).

## 6.2. Đóng góp của đề tài

- Xây dựng được một **pipeline hoàn chỉnh, tự động** từ dữ liệu thô đến kết quả
  đánh giá (`python main.py all`), có tổ chức mã nguồn rõ ràng, chú thích đầy đủ.
- Chỉ ra và xử lý đúng vấn đề **dữ liệu trùng lặp gây rò rỉ dữ liệu** — một lỗi
  phổ biến dễ làm kết quả bị "ảo".
- Cung cấp bộ công cụ **đánh giá và trực quan hóa** đầy đủ (các chỉ số, Confusion
  Matrix, ROC, biểu đồ so sánh) phục vụ phân tích.

## 6.3. Hạn chế

- **Kích thước dữ liệu nhỏ:** sau khi loại trùng chỉ còn 302 bản ghi, hạn chế khả
  năng học của mạng nơ-ron và làm kết quả dao động giữa các cấu hình.
- **Dữ liệu cũ (từ 1988)** và có một số giá trị bất thường về mã hoá (`ca = 4`,
  `thal = 0`), có thể không phản ánh đầy đủ thực tế hiện nay.
- Mô hình mới dừng ở mức **sàng lọc/hỗ trợ**, chưa thể thay thế chẩn đoán y khoa.

## 6.4. Hướng phát triển

- **Mở rộng dữ liệu:** thu thập thêm dữ liệu thực tế, cập nhật và đa dạng hơn để
  tăng độ chính xác và khả năng tổng quát.
- **Thử nghiệm mô hình & kỹ thuật khác:** so sánh ANN với các mô hình như Random
  Forest, XGBoost; áp dụng k-fold cross-validation, cân bằng dữ liệu, tinh chỉnh
  siêu tham số tự động (Grid/Random Search).
- **Giải thích mô hình (Explainable AI):** dùng SHAP/feature importance để giải
  thích yếu tố nào ảnh hưởng đến dự đoán — quan trọng trong y tế.
- **Triển khai ứng dụng:** đóng gói mô hình thành web/app cho phép nhập chỉ số và
  trả về cảnh báo nguy cơ, hỗ trợ bác sĩ và người dùng.

## 6.5. Lời kết

Qua đề tài, nhóm đã vận dụng được kiến thức môn Trí tuệ nhân tạo vào một bài toán
y tế thực tiễn, nắm vững quy trình xây dựng – huấn luyện – đánh giá một mô hình
học máy, đồng thời rèn luyện kỹ năng làm việc nhóm và quản lý mã nguồn bằng Git.
