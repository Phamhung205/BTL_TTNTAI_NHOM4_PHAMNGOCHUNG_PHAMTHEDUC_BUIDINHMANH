# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

> Người thực hiện: **Bùi Đình Mạnh**

---

## 2.1. Trí tuệ nhân tạo (Artificial Intelligence – AI)

**Trí tuệ nhân tạo** là lĩnh vực của khoa học máy tính nghiên cứu việc tạo ra các
hệ thống có khả năng thực hiện những nhiệm vụ vốn đòi hỏi trí thông minh của con
người, như nhận dạng hình ảnh, hiểu ngôn ngữ, ra quyết định và dự đoán. AI bao
gồm nhiều nhánh, trong đó **Học máy (Machine Learning)** là nhánh phát triển mạnh
mẽ và được ứng dụng rộng rãi nhất hiện nay.

## 2.2. Học máy (Machine Learning – ML)

**Học máy** là phương pháp giúp máy tính "học" quy luật từ dữ liệu thay vì được
lập trình cứng bằng các quy tắc cố định. Có ba nhóm chính:

- **Học có giám sát (Supervised Learning):** học từ dữ liệu đã gán nhãn để dự
  đoán nhãn cho dữ liệu mới (phân loại, hồi quy). *Bài toán của đề tài thuộc nhóm
  này — phân loại nhị phân.*
- **Học không giám sát (Unsupervised Learning):** tìm cấu trúc ẩn trong dữ liệu
  chưa gán nhãn (phân cụm, giảm chiều).
- **Học tăng cường (Reinforcement Learning):** học qua tương tác và phần thưởng.

Bài toán đánh giá nguy cơ bệnh tim là bài toán **phân loại nhị phân** (binary
classification): đầu ra là 1 (có nguy cơ) hoặc 0 (không có nguy cơ).

## 2.3. Mạng nơ-ron nhân tạo (Artificial Neural Network – ANN)

**Mạng nơ-ron nhân tạo** là mô hình tính toán mô phỏng cách hoạt động của các
nơ-ron thần kinh trong não người. ANN gồm các **nơ-ron (neuron)** được tổ chức
thành nhiều **tầng (layer)**:

- **Tầng đầu vào (Input Layer):** nhận vector đặc trưng (ở đề tài là 13 đặc trưng).
- **Các tầng ẩn (Hidden Layers):** trích xuất và biến đổi đặc trưng; càng nhiều
  tầng/nơ-ron, mạng càng học được quan hệ phức tạp.
- **Tầng đầu ra (Output Layer):** đưa ra kết quả dự đoán (1 nơ-ron với hàm
  Sigmoid cho phân loại nhị phân).

Mỗi kết nối giữa hai nơ-ron có một **trọng số (weight)**, mỗi nơ-ron có thêm một
**độ lệch (bias)**. Giá trị tại một nơ-ron được tính bằng:

```
z = (w1·x1 + w2·x2 + ... + wn·xn) + b
a = f(z)
```

trong đó `f` là **hàm kích hoạt**. Quá trình tính toán từ đầu vào đến đầu ra gọi
là **lan truyền tiến (Forward Propagation)**.

## 2.4. Hàm kích hoạt (Activation Function)

Hàm kích hoạt đưa tính **phi tuyến** vào mạng, giúp ANN học được các quan hệ phức
tạp. Hai hàm dùng trong đề tài:

- **ReLU (Rectified Linear Unit):** `f(z) = max(0, z)`. Dùng ở các tầng ẩn vì
  tính toán nhanh, giảm hiện tượng *vanishing gradient* so với sigmoid/tanh.
- **Sigmoid:** `f(z) = 1 / (1 + e^(-z))`, cho giá trị trong khoảng (0, 1). Dùng ở
  tầng đầu ra để biểu diễn **xác suất** thuộc lớp "có nguy cơ"; ngưỡng 0.5 dùng
  để quy về nhãn 0/1.

## 2.5. Hàm mất mát (Loss Function)

**Hàm mất mát** đo độ sai lệch giữa dự đoán và nhãn thực tế. Với phân loại nhị
phân, dùng **Binary Crossentropy (Log Loss)**:

```
L = -(1/N) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

trong đó `y` là nhãn thực, `ŷ` là xác suất dự đoán. Giá trị loss càng nhỏ thì mô
hình dự đoán càng chính xác.

## 2.6. Lan truyền ngược (Backpropagation)

**Backpropagation** là thuật toán cốt lõi để huấn luyện ANN. Ý tưởng:

1. **Forward:** tính đầu ra và giá trị loss.
2. **Backward:** tính **gradient** (đạo hàm) của loss theo từng trọng số, lan
   truyền ngược từ tầng đầu ra về tầng đầu vào theo quy tắc chuỗi (chain rule).
3. **Cập nhật trọng số** theo hướng giảm loss (gradient descent):
   `w ← w − η·(∂L/∂w)`, với `η` là **tốc độ học (learning rate)**.

Quá trình lặp lại qua nhiều **epoch** (lượt duyệt toàn bộ dữ liệu) cho đến khi mô
hình hội tụ.

## 2.7. Bộ tối ưu (Optimizer) – Adam

**Adam (Adaptive Moment Estimation)** là bộ tối ưu cải tiến của gradient descent,
kết hợp **momentum** và **điều chỉnh tốc độ học thích nghi** cho từng tham số.
Adam hội tụ nhanh, ổn định và ít phải tinh chỉnh thủ công, nên được chọn cho mô
hình. Tốc độ học sử dụng là `0.001`, kích thước batch `32`.

## 2.8. Quá khớp (Overfitting) và kỹ thuật chống quá khớp

**Overfitting** xảy ra khi mô hình học "thuộc lòng" dữ liệu train mà kém tổng quát
trên dữ liệu mới (loss train thấp nhưng loss validation cao). Các kỹ thuật chống:

- **Early Stopping:** theo dõi loss trên tập validation, dừng huấn luyện khi loss
  không cải thiện sau một số epoch (patience) và giữ lại trọng số tốt nhất.
- **Regularization (L2):** thêm thành phần phạt vào loss để hạn chế trọng số lớn.
- **Dropout:** ngẫu nhiên "tắt" một phần nơ-ron khi huấn luyện (tuỳ chọn).
- **Chia tập dữ liệu hợp lý** và **loại bỏ dữ liệu trùng** để tránh rò rỉ dữ liệu.

## 2.9. Các chỉ số đánh giá mô hình phân loại

Dựa trên **ma trận nhầm lẫn (Confusion Matrix)** với 4 đại lượng:
TP (dự đoán đúng có bệnh), TN (đúng không bệnh), FP (báo nhầm có bệnh),
FN (bỏ sót ca có bệnh).

- **Accuracy** = (TP + TN) / Tổng — tỷ lệ dự đoán đúng tổng thể.
- **Precision** = TP / (TP + FP) — độ chính xác của dự đoán "có bệnh".
- **Recall (Sensitivity)** = TP / (TP + FN) — khả năng phát hiện ca có bệnh
  (đặc biệt quan trọng trong y tế, vì bỏ sót bệnh nhân rất nguy hiểm).
- **F1-Score** = 2·(Precision·Recall)/(Precision+Recall) — trung bình điều hòa,
  cân bằng giữa Precision và Recall.
- **ROC-AUC:** diện tích dưới đường ROC, đo khả năng phân tách hai lớp ở mọi
  ngưỡng (0.5 = ngẫu nhiên, 1.0 = hoàn hảo).

Các chỉ số này được dùng để đánh giá và so sánh các mô hình ở **Chương 5**.
