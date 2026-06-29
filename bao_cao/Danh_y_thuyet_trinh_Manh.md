# Dàn ý thuyết trình (≈3 phút) — Phần của Bùi Đình Mạnh

> Phạm vi trình bày: đánh giá mô hình, biểu đồ trực quan, so sánh cấu hình, kết luận.
> Dùng cùng với các hình trong `results/` (chiếu slide hoặc mở trực tiếp file ảnh).

---

## 0. Mở đầu (5 giây)
> "Sau khi anh/bạn [Hùng] đã trình bày kiến trúc và quá trình huấn luyện mô hình,
> em xin trình bày phần **đánh giá kết quả thực nghiệm** và **kết luận** của đề tài."

## 1. Cách đánh giá (20 giây)
- Mô hình được kiểm tra trên **tập test gồm 61 mẫu** (20% dữ liệu, chưa từng thấy
  khi huấn luyện).
- Dùng 5 chỉ số chuẩn: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**.
- *Nhấn mạnh:* trong y tế, **Recall** quan trọng hơn Accuracy đơn thuần — vì bỏ
  sót một ca bệnh (False Negative) nguy hiểm hơn báo nhầm.

## 2. Quá trình huấn luyện (20 giây) — chiếu `accuracy_curve.png`, `loss_curve.png`
- Mô hình hội tụ ổn định, **dừng sớm ở epoch 38** nhờ Early Stopping.
- Đường train và validation bám sát nhau lúc đầu, sau đó validation chững lại
  đúng lúc mô hình bắt đầu có dấu hiệu quá khớp → Early Stopping hoạt động đúng
  như kỳ vọng.

## 3. Kết quả chính (30 giây) — chiếu bảng chỉ số + `confusion_matrix.png`
- **Accuracy ≈ 0.74, F1-Score ≈ 0.76, ROC-AUC ≈ 0.85.**
- Confusion Matrix: bắt đúng **25/33 ca có bệnh** và **20/28 ca không bệnh**;
  số ca bỏ sót và báo nhầm đều là 8, ở mức cân bằng.
- ROC-AUC 0.85 cho thấy mô hình **phân tách hai lớp tốt**, vượt xa mức ngẫu
  nhiên (0.5).

## 4. So sánh 4 cấu hình (40 giây) — chiếu `model_comparison.png` + bảng
- Thử nghiệm 4 kiến trúc khác nhau (số tầng ẩn, learning rate, batch size).
- **ANN-3 (64-32-16, lr=0.0001, batch=16)** cho **F1 và Recall cao nhất**
  (F1 ≈ 0.79, Recall ≈ 0.82) → bắt được nhiều ca bệnh nhất, phù hợp mục tiêu y tế.
- ANN-4 (mạng lớn nhất) có ROC-AUC cao nhất nhưng F1 thấp hơn → mạng quá lớn
  **không** giúp ích thêm trên bộ dữ liệu nhỏ, thậm chí dễ quá khớp.
- → Chọn **ANN-3** làm cấu hình đề xuất cuối cùng.

## 5. Hạn chế & hướng phát triển (30 giây)
- Hạn chế lớn nhất: **dữ liệu sau khi loại trùng chỉ còn 302 bản ghi** — khá nhỏ
  cho một mạng nơ-ron, khiến kết quả có thể dao động.
- Hướng phát triển: thu thập thêm dữ liệu thực tế, so sánh với Random
  Forest/XGBoost, áp dụng cross-validation, và dùng SHAP để giải thích mô hình
  (quan trọng trong y tế — bác sĩ cần biết "vì sao" mô hình dự đoán như vậy).

## 6. Kết luận (15 giây)
> "Tổng kết lại, mô hình ANN đạt kết quả khả thi (ROC-AUC ≈ 0.85) cho bài toán
> sàng lọc nguy cơ bệnh tim. Đây là một công cụ **hỗ trợ**, chưa thay thế chẩn
> đoán y khoa, nhưng cho thấy hướng tiếp cận AI trong y tế là khả thi và đáng
> để mở rộng trong tương lai. Em xin hết phần trình bày, cảm ơn cô/thầy và các
> bạn đã theo dõi."

---

### Lưu ý khi trình bày
- Nếu được hỏi *"vì sao không dùng TensorFlow như thiết kế gốc?"*: trả lời ngắn
  gọn — "máy thực nghiệm bị Windows Smart App Control chặn TensorFlow, nhóm
  dùng `MLPClassifier` của scikit-learn với **cùng kiến trúc và siêu tham số**
  để đảm bảo kết quả tương đương; file Keras gốc (`model.py`, `train.py`) vẫn
  được giữ trong source code."
- Nếu được hỏi *"tại sao không chọn mô hình có Accuracy cao nhất?"*: trả lời —
  "vì trong y tế, bỏ sót bệnh nhân (Recall thấp) nguy hiểm hơn báo nhầm, nên
  nhóm ưu tiên F1/Recall cao khi chọn mô hình tốt nhất, không chỉ riêng Accuracy."
