# Hướng dẫn dự án: Đánh giá nguy cơ mắc bệnh tim bằng ANN

Tài liệu này ghi lại lộ trình thực hiện phần thiết kế & huấn luyện mô hình ANN.
Đánh dấu ✅ vào mục đã xong để cả nhóm theo dõi tiến độ.

---

## 0. Chuẩn bị môi trường (ĐÃ XONG ✅)

- [x] Cài Python 3.12
- [x] Tạo môi trường ảo `venv`
- [x] Kích hoạt venv: `venv\Scripts\activate`
- [x] Cài thư viện: tensorflow, scikit-learn, pandas, numpy, matplotlib, jupyter
- [x] Chọn interpreter venv trong VS Code (Ctrl+Shift+P → Python: Select Interpreter)

**Lưu ý mỗi lần mở lại dự án:** phải kích hoạt lại venv trước khi chạy code.
Mở Terminal trong VS Code rồi gõ:
```
venv\Scripts\activate
```
Nếu thấy `(venv)` ở đầu dòng là đúng.

---

## Cấu trúc thư mục mục tiêu

```
BTL TTNTAI/
├── venv/                          (môi trường ảo - không đụng vào)
├── data/
│   └── heart.csv                  (dữ liệu - bạn cùng nhóm cung cấp)
├── src/
│   ├── model.py                   ✅ ĐÃ XONG - kiến trúc ANN
│   ├── data_loader.py             (đọc & chia dữ liệu train/test)
│   └── train.py                   (huấn luyện, lưu model)
├── saved_models/
│   ├── best_model.h5              (model tốt nhất - tự sinh khi train)
│   └── ann_model.h5               (model cuối cùng - tự sinh khi train)
├── notebook/
│   ├── model_experiments.ipynb    (thử nghiệm các kiến trúc)
│   └── hyperparameter_tuning.ipynb(so sánh siêu tham số)
└── HUONG_DAN.md                   (file này)
```

**Việc cần làm ngay:** tạo các thư mục còn thiếu. Trong Explorer của VS Code,
nhấn chuột phải vào `BTL TTNTAI` → New Folder, tạo lần lượt:
`data`, `saved_models`, `notebook`.

---

## PHẦN 1 — Thiết kế mô hình ANN (ĐÃ XONG ✅)

- [x] Viết `src/model.py` với hàm `build_model()`
- [x] Kiến trúc: Input(13) → Dense(64) → Dense(32) → Dense(16) → Dense(1, sigmoid)
- [x] Compile: Adam + BinaryCrossentropy + các metric (accuracy, AUC, precision, recall)
- [x] Chạy thử `python src/model.py` → in ra bảng, Total params: 3,521

---

## PHẦN 2 — Huấn luyện mô hình (TIẾP THEO 👉)

Mục tiêu: viết `src/train.py` để huấn luyện và lưu model.

Các bước:
1. Đọc dữ liệu đã tiền xử lý (file CSV từ bạn cùng nhóm).
2. Gọi `build_model()` từ `model.py`.
3. Cấu hình huấn luyện:
   - Loss: BinaryCrossentropy
   - Optimizer: Adam
   - Learning rate gợi ý: 0.001
   - Batch size gợi ý: 32
   - Epochs gợi ý: 100 (sẽ tự dừng sớm nhờ EarlyStopping)
4. Dùng callback:
   - **EarlyStopping**: dừng khi val_loss không cải thiện → chống overfitting
   - **ModelCheckpoint**: tự lưu `best_model.h5` mỗi khi val tốt hơn
5. Sau khi train xong: lưu thêm `ann_model.h5` (model ở epoch cuối).
6. Vẽ biểu đồ accuracy/loss theo epoch để đưa vào báo cáo.

**Cần cho phần này:** file dữ liệu hoặc danh sách tên các cột (đặc trưng + nhãn).
Nếu chưa có, sẽ tạo dữ liệu mẫu để chạy thử pipeline trước.

---

## PHẦN 3 — Thử nghiệm siêu tham số (SAU PHẦN 2)

Viết 2 notebook chạy thử 4 cấu hình rồi so sánh:

| Thí nghiệm | Hidden Layers | Learning Rate | Batch Size |
|---|---|---|---|
| 1 | 32-16        | 0.001  | 32 |
| 2 | 64-32        | 0.001  | 32 |
| 3 | 64-32-16     | 0.0001 | 16 |
| 4 | 128-64-32    | 0.001  | 64 |

Mỗi thí nghiệm ghi lại: accuracy, loss (và AUC, precision, recall).
Tổng hợp thành 1 bảng → chọn cấu hình tốt nhất.

**Cách chạy notebook trong VS Code:**
- Mở file `.ipynb`, VS Code tự nhận.
- Góc trên phải chọn Kernel → chọn venv.
- Chạy từng ô bằng nút ▶ bên trái ô, hoặc Shift+Enter.

---

## PHẦN 4 — Hỗ trợ viết báo cáo (CUỐI CÙNG)

Dựa trên kết quả thật, viết nội dung:
- **Chương 4 — Mô hình ANN:** trình bày kiến trúc, lý do chọn từng layer,
  hàm kích hoạt, hàm mất mát, bộ tối ưu.
- **Chương 5 — Kết quả thực nghiệm:** quá trình huấn luyện, biểu đồ accuracy/loss,
  bảng so sánh 4 thí nghiệm, kết luận chọn mô hình.

---

## Mẹo dùng VS Code nhanh

| Thao tác | Cách làm |
|---|---|
| Mở Terminal | Ctrl + ` (phím dấu huyền, dưới Esc) |
| Chạy file Python | Gõ `python src/ten_file.py` ở Terminal |
| Kích hoạt venv | `venv\Scripts\activate` |
| Chọn interpreter | Ctrl+Shift+P → "Python: Select Interpreter" |
| Lưu file | Ctrl + S |

## Lưu ý về cảnh báo TensorFlow (KHÔNG phải lỗi)

Mỗi lần chạy, TensorFlow in vài dòng như:
- `WARNING: All log messages before absl::InitializeLog()...`
- `oneDNN custom operations are on...`
- `TensorFlow GPU support is not available on native Windows...`

Đây đều là **thông báo bình thường**, không phải lỗi. Bài này chạy bằng CPU
là đủ. Chỉ cần lo lắng khi thấy chữ **Error** hoặc **Traceback** màu đỏ.
