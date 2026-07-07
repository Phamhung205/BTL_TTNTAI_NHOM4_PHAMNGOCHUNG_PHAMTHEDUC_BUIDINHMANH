# CHƯƠNG 4. THIẾT KẾ VÀ HUẤN LUYỆN MÔ HÌNH ANN

> Người thực hiện: **Phạm Ngọc Hùng** (Nhóm trưởng — Model Engineer)
> Mã nguồn: `src/model.py`, `src/data_loader.py`, `src/train.py`
> (bản scikit-learn tương đương: `src/train_ann.py`)

---

## 4.1. Bài toán và yêu cầu đối với mô hình

Sau khi dữ liệu đã được tiền xử lý ở Chương 3 (302 bản ghi sạch, 13 đặc trưng đã
chuẩn hoá, nhãn `target` 0/1), nhiệm vụ của chương này là thiết kế một **mạng
nơ-ron nhân tạo (ANN)** giải bài toán **phân loại nhị phân**:

- **Đầu vào:** vector 13 đặc trưng y tế của một bệnh nhân.
- **Đầu ra:** xác suất bệnh nhân **có nguy cơ mắc bệnh tim** (giá trị trong
  khoảng 0–1; ngưỡng 0.5 để quy về nhãn 0/1).

Yêu cầu thiết kế: mô hình đủ sức học quan hệ phi tuyến giữa các chỉ số y tế,
nhưng **không quá lớn** so với lượng dữ liệu khiêm tốn (302 mẫu) để tránh quá
khớp (overfitting).

## 4.2. Kiến trúc mạng

Mô hình dùng kiến trúc **Sequential** (các tầng xếp tuần tự), cài đặt trong
`src/model.py` bằng hàm `build_model()`:

```
Input(13) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)
```

| Tầng | Số nơ-ron | Hàm kích hoạt | Vai trò |
|---|---|---|---|
| Input | 13 | — | Nhận vector 13 đặc trưng đã chuẩn hoá |
| Hidden 1 | 64 | ReLU | Trích xuất đặc trưng tổng quát |
| Hidden 2 | 32 | ReLU | Nén và kết hợp đặc trưng |
| Hidden 3 | 16 | ReLU | Cô đọng thông tin trước khi ra quyết định |
| Output | 1 | Sigmoid | Xác suất "có nguy cơ" trong [0, 1] |

**Lý do lựa chọn:**

- **Số nơ-ron giảm dần (64 → 32 → 16):** mạng "nén" dần thông tin, học đặc trưng
  từ tổng quát đến cô đọng — kiến trúc hình phễu phổ biến cho dữ liệu bảng.
- **ReLU ở tầng ẩn:** tính toán nhanh, giảm hiện tượng *vanishing gradient* so
  với sigmoid/tanh (đã trình bày ở Chương 2).
- **Sigmoid ở tầng ra:** đầu ra là xác suất, phù hợp phân loại nhị phân và hàm
  mất mát Binary Crossentropy.
- Hàm `build_model()` được viết **tham số hoá** (`hidden_units`, `dropout_rate`,
  `learning_rate`) để dễ dàng thử nghiệm nhiều cấu hình ở Chương 5.

Tổng số tham số học của mô hình: 13×64 + 64 + 64×32 + 32 + 32×16 + 16 + 16×1 + 1
= **3 521 tham số** — nhỏ gọn, phù hợp với 302 mẫu dữ liệu.

## 4.3. Cấu hình huấn luyện

Cấu hình được khai báo tập trung ở đầu `src/train.py`:

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Hàm mất mát | **Binary Crossentropy** | Chuẩn cho phân loại nhị phân |
| Bộ tối ưu | **Adam** | Hội tụ nhanh, ổn định |
| Learning rate | 0.001 | Giá trị mặc định khuyến nghị của Adam |
| Batch size | 32 | Cân bằng tốc độ và độ ổn định gradient |
| Epochs tối đa | 100 | Early Stopping sẽ dừng sớm |
| Validation split | 20% của tập train | Theo dõi overfitting trong lúc huấn luyện |
| Metrics theo dõi | Accuracy, AUC, Precision, Recall | Khai báo khi compile |

## 4.4. Quy trình huấn luyện

Quy trình trong `src/train.py` gồm các bước:

1. **Đọc và chia dữ liệu** qua `data_loader.load_data()`: đọc file
   `data/heart.csv` (đã tiền xử lý ở Chương 3), tách X/y và chia **80/20** với
   `stratify=y` (giữ tỷ lệ hai lớp cân bằng giữa train và test),
   `random_state=42` (kết quả lặp lại được).
2. **Dựng mô hình** bằng `build_model(input_dim=13)`.
3. **Huấn luyện** với hai callback quan trọng:
   - **EarlyStopping** (`monitor="val_loss"`, `patience=10`,
     `restore_best_weights=True`): nếu `val_loss` không cải thiện sau 10 epoch
     liên tiếp thì dừng và khôi phục lại bộ trọng số tốt nhất — chống overfitting
     và tiết kiệm thời gian.
   - **ModelCheckpoint** (`save_best_only=True`): tự động lưu mô hình có
     `val_loss` thấp nhất ra `saved_models/best_model.h5`.
4. **Lưu kết quả:** mô hình cuối (`ann_model.h5`), lịch sử huấn luyện
   (`results/history.json` — phục vụ vẽ biểu đồ ở Chương 5) và biểu đồ
   accuracy/loss.
5. **Đánh giá nhanh** trên tập test bằng `model.evaluate()`.

## 4.5. Bản cài đặt tương đương bằng scikit-learn

Do máy thực nghiệm của nhóm bị **Windows Smart App Control** chặn thư viện
TensorFlow (chi tiết ở Chương 5), nhóm cài đặt thêm một bản huấn luyện **tương
đương** bằng scikit-learn trong `src/train_ann.py`:

- Dùng `MLPClassifier` với **cùng kiến trúc** `hidden_layer_sizes=(64, 32, 16)`,
  `activation="relu"`, solver **Adam**, `learning_rate_init=0.001`,
  `batch_size=32` — khớp từng siêu tham số với bản Keras.
- Huấn luyện bằng vòng lặp `partial_fit` theo từng epoch, tự tính
  `log_loss`/`accuracy` trên train và validation sau mỗi epoch (tương đương
  `history` của Keras) và tự cài **Early Stopping** (patience = 10 theo
  val_loss, giữ lại bản sao mô hình tốt nhất).
- Kết quả lưu tại `saved_models/ann_sklearn.joblib` và `results/history.json`
  — **cùng định dạng** với bản Keras, nên toàn bộ khâu đánh giá/trực quan hóa
  (Chương 5) dùng chung không cần sửa.

Về mặt toán học, hai bản cài đặt cùng tối ưu hàm mất mát log loss (Binary
Crossentropy) bằng Adam trên cùng kiến trúc mạng, nên kết quả tương đương nhau;
khác biệt chỉ nằm ở thư viện thực thi.

## 4.6. Kết quả huấn luyện

Mô hình hội tụ và **dừng sớm tại epoch 38** nhờ Early Stopping. Đường
accuracy/loss theo epoch, các chỉ số chi tiết trên tập test và so sánh giữa các
cấu hình siêu tham số được trình bày đầy đủ trong **Chương 5 — Kết quả thực
nghiệm**.

## 4.7. Kết luận chương

Chương 4 đã hoàn thành việc thiết kế và huấn luyện mô hình ANN cho bài toán đánh
giá nguy cơ bệnh tim: kiến trúc 13 → 64 → 32 → 16 → 1 (ReLU/Sigmoid), huấn luyện
bằng Adam + Binary Crossentropy với Early Stopping và ModelCheckpoint, mã nguồn
tham số hoá để dễ thử nghiệm cấu hình. Mô hình đã sẵn sàng cho khâu đánh giá và
so sánh ở Chương 5.
