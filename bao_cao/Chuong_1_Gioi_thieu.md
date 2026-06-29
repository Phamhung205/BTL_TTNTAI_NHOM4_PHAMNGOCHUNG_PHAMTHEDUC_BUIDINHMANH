# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

> Người thực hiện: **Bùi Đình Mạnh**

---

## 1.1. Lý do chọn đề tài

Bệnh tim mạch là một trong những nguyên nhân gây tử vong hàng đầu trên thế giới
cũng như tại Việt Nam. Theo Tổ chức Y tế Thế giới (WHO), mỗi năm có hàng chục
triệu người tử vong do các bệnh lý tim mạch, và phần lớn các trường hợp có thể
được phòng ngừa hoặc điều trị hiệu quả hơn nếu được **phát hiện sớm nguy cơ**.

Tuy nhiên, việc chẩn đoán nguy cơ bệnh tim truyền thống phụ thuộc nhiều vào kinh
nghiệm của bác sĩ và các xét nghiệm chuyên sâu, tốn thời gian và chi phí. Trong
bối cảnh đó, **Trí tuệ nhân tạo (AI)** — đặc biệt là các mô hình học máy và mạng
nơ-ron nhân tạo — mở ra hướng tiếp cận mới: xây dựng hệ thống **hỗ trợ sàng lọc,
dự đoán nguy cơ** dựa trên các thông số y tế cơ bản, giúp cảnh báo sớm và hỗ trợ
ra quyết định cho bác sĩ.

Xuất phát từ nhu cầu thực tế đó, nhóm chọn đề tài **"Đánh giá nguy cơ mắc bệnh
tim bằng Mạng nơ-ron nhân tạo (ANN)"** nhằm vận dụng kiến thức môn Trí tuệ nhân
tạo vào một bài toán y tế có ý nghĩa thực tiễn.

## 1.2. Mục tiêu nghiên cứu

- Tìm hiểu và áp dụng quy trình xây dựng một mô hình học máy hoàn chỉnh: từ thu
  thập, tiền xử lý dữ liệu đến huấn luyện, đánh giá và so sánh mô hình.
- Thiết kế và huấn luyện mô hình **Mạng nơ-ron nhân tạo (ANN)** phân loại nhị
  phân nguy cơ mắc bệnh tim.
- Đánh giá hiệu năng mô hình bằng các chỉ số chuẩn (Accuracy, Precision, Recall,
  F1-Score, ROC-AUC) và trực quan hóa kết quả.
- Thử nghiệm nhiều cấu hình siêu tham số khác nhau để tìm ra mô hình tốt nhất.

## 1.3. Đối tượng và phạm vi nghiên cứu

- **Đối tượng:** bài toán phân loại nguy cơ mắc bệnh tim dựa trên dữ liệu bảng
  (tabular data) gồm các chỉ số y tế của bệnh nhân.
- **Dữ liệu:** bộ **Heart Disease Dataset** trên Kaggle
  (`johnsmith88/heart-disease-dataset`), gồm 1025 bản ghi với 13 đặc trưng đầu
  vào và 1 nhãn mục tiêu.
- **Phạm vi:** tập trung vào mô hình ANN cho bài toán phân loại nhị phân; không
  đi sâu vào các mô hình học sâu phức tạp khác (CNN, RNN) vì dữ liệu ở dạng bảng.

## 1.4. Phương pháp nghiên cứu

- **Phương pháp thu thập & xử lý dữ liệu:** tải dữ liệu từ Kaggle, kiểm tra chất
  lượng (giá trị thiếu, trùng lặp, ngoại lệ), làm sạch và chuẩn hoá.
- **Phương pháp mô hình hóa:** sử dụng Mạng nơ-ron nhân tạo (ANN) nhiều tầng với
  hàm kích hoạt ReLU/Sigmoid, huấn luyện bằng thuật toán lan truyền ngược
  (Backpropagation) và bộ tối ưu Adam.
- **Phương pháp đánh giá:** chia dữ liệu train/test, đánh giá bằng nhiều chỉ số
  và trực quan hóa (Confusion Matrix, ROC Curve), so sánh các cấu hình.

## 1.5. Công cụ và môi trường

- Ngôn ngữ: **Python 3**.
- Thư viện: pandas, numpy, scikit-learn, matplotlib, seaborn (và TensorFlow/Keras
  cho bản mô hình gốc).
- Công cụ: VS Code, Jupyter Notebook, Git/GitHub.

## 1.6. Cấu trúc báo cáo

Báo cáo gồm 6 chương:

- **Chương 1 — Giới thiệu đề tài:** bối cảnh, mục tiêu, phạm vi và phương pháp.
- **Chương 2 — Cơ sở lý thuyết:** AI, Machine Learning, ANN, Backpropagation,
  hàm kích hoạt, hàm mất mát, bộ tối ưu và các chỉ số đánh giá.
- **Chương 3 — Dữ liệu và tiền xử lý:** giới thiệu bộ dữ liệu, làm sạch, feature
  engineering và phân tích khám phá (EDA).
- **Chương 4 — Thiết kế và huấn luyện mô hình ANN:** kiến trúc mạng, cấu hình
  huấn luyện.
- **Chương 5 — Kết quả thực nghiệm:** kết quả đánh giá, biểu đồ, so sánh các
  cấu hình.
- **Chương 6 — Kết luận và hướng phát triển.**
