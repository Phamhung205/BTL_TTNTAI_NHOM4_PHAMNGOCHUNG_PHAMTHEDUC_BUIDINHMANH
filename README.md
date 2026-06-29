# Đánh giá nguy cơ mắc bệnh tim bằng Mạng nơ-ron nhân tạo (ANN)

> Bài tập lớn môn **Trí tuệ nhân tạo** — Nhóm 4
> Đề tài 4: *Heart Disease Risk Assessment*

Dự án xây dựng mô hình **Mạng nơ-ron nhân tạo (ANN)** phân loại bệnh nhân thuộc
nhóm **có nguy cơ** hay **không có nguy cơ** mắc bệnh tim, dựa trên các thông số
y tế sàng lọc (tuổi, giới tính, huyết áp, cholesterol, điện tâm đồ, ...).

---

## 1. Thành viên nhóm & phân công

| Thành viên | Vai trò | Phụ trách |
|---|---|---|
| **Phạm Ngọc Hùng** (Nhóm trưởng) | Model Engineer | Thiết kế & huấn luyện ANN, tối ưu siêu tham số, Chương 4 |
| **Phạm Thế Đức** | Data Engineer & Data Analyst | Tiền xử lý dữ liệu, feature engineering, EDA, Chương 3 |
| **Bùi Đình Mạnh** | Evaluation & Documentation | Đánh giá mô hình, trực quan hóa, so sánh, Chương 1-2-5-6, README |

---

## 2. Bộ dữ liệu

- **Nguồn:** Kaggle — `johnsmith88/heart-disease-dataset`.
- **Quy mô gốc:** 1025 bản ghi, 14 thuộc tính (13 đặc trưng + nhãn `target`).
- **Sau làm sạch:** loại 723 bản ghi **trùng lặp** → còn **302 bản ghi duy nhất**
  (chi tiết trong Chương 3).
- **Nhãn:** `target = 1` (có nguy cơ bệnh tim) / `target = 0` (không).

---

## 3. Cấu trúc thư mục

```
.
├── data/
│   ├── raw/heart.csv              # dữ liệu thô từ Kaggle
│   ├── processed/                 # train.csv, test.csv, heart_clean.csv
│   └── heart.csv                  # dữ liệu đã xử lý (1 file tổng cho train.py)
├── src/
│   ├── data_preprocessing.py      # (Đức) làm sạch dữ liệu
│   ├── feature_engineering.py     # (Đức) mã hoá, chuẩn hoá, chia train/test
│   ├── data_loader.py             # (Hùng) đọc & chia dữ liệu
│   ├── model.py                   # (Hùng) kiến trúc ANN bằng Keras/TensorFlow
│   ├── train.py                   # (Hùng) huấn luyện bản Keras
│   ├── train_ann.py               # (Mạnh) huấn luyện ANN bằng scikit-learn
│   ├── evaluate.py                # (Mạnh) tính Accuracy/Precision/Recall/F1/AUC
│   ├── visualization.py           # (Mạnh) vẽ accuracy/loss/confusion/ROC
│   └── compare_models.py          # (Mạnh) so sánh 4 cấu hình siêu tham số
├── notebook/
│   ├── data_exploration.ipynb     # (Đức) EDA
│   ├── model_experiments.ipynb    # (Hùng)
│   └── hyperparameter_tuning.ipynb# (Hùng)
├── results/                       # (Mạnh) biểu đồ + chỉ số đánh giá (tự sinh)
├── saved_models/                  # model đã huấn luyện (tự sinh)
├── bao_cao/                       # báo cáo Chương 1–6 (md + tex) + figures
├── main.py                        # chương trình chính (chạy toàn bộ pipeline)
├── requirements.txt               # thư viện đầy đủ (gồm TensorFlow)
└── requirements_sklearn.txt       # thư viện bản scikit-learn (không cần TF)
```

---

## 4. Cài đặt

```bash
# 1) Lấy mã nguồn
git clone https://github.com/Phamhung205/BTL_TTNTAI_NHOM4_PHAMNGOCHUNG_PHAMTHEDUC_BUIDINHMANH.git
cd BTL_TTNTAI_NHOM4_PHAMNGOCHUNG_PHAMTHEDUC_BUIDINHMANH

# 2) Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

# 3) Cài thư viện
pip install -r requirements.txt          # bản đầy đủ (có TensorFlow)
# HOAC neu may khong cai duoc TensorFlow:
pip install -r requirements_sklearn.txt  # bản scikit-learn
```

> **Lưu ý môi trường:** Mô hình được thiết kế bằng **Keras/TensorFlow**
> (`src/model.py`, `src/train.py`). Trên máy không cài được TensorFlow
> (ví dụ Python 3.13/3.14, hoặc bị **Windows Smart App Control** chặn DLL),
> hãy dùng `requirements_sklearn.txt` và bản huấn luyện tương đương
> `src/train_ann.py` (scikit-learn, cùng kiến trúc 64-32-16). Khuyến nghị
> **Python 3.12** cho ổn định.

---

## 5. Cách chạy

### Cách 1 — chạy toàn bộ pipeline (khuyến nghị)
```bash
python main.py all
```
Lần lượt: tiền xử lý → feature engineering → huấn luyện → đánh giá →
trực quan hóa → so sánh mô hình.

### Cách 2 — menu tương tác
```bash
python main.py
```

### Cách 3 — chạy từng bước
```bash
python src/data_preprocessing.py     # 1. làm sạch dữ liệu
python src/feature_engineering.py    # 2. chuẩn hoá + chia train/test
python src/train_ann.py              # 3. huấn luyện ANN (scikit-learn)
python src/evaluate.py               # 4. đánh giá mô hình
python src/visualization.py          # 5. vẽ biểu đồ
python src/compare_models.py         # 6. so sánh 4 cấu hình
```

> Trên máy có TensorFlow, có thể huấn luyện bản Keras gốc: `python src/train.py`.

---

## 6. Kết quả tóm tắt

Mô hình ANN (kiến trúc 64-32-16) trên **tập test (61 mẫu)**:

| Chỉ số | Giá trị |
|---|---|
| Accuracy | 0.738 |
| Precision | 0.758 |
| Recall | 0.758 |
| F1-Score | 0.758 |
| ROC-AUC | 0.850 |

**So sánh 4 cấu hình siêu tham số** (xem `results/model_comparison.csv`):

| Mô hình | Hidden | LR | Batch | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| ANN-1 | 32-16 | 0.001 | 32 | 0.771 | 0.788 | 0.859 |
| ANN-2 | 64-32 | 0.001 | 32 | 0.771 | 0.788 | 0.843 |
| **ANN-3** | 64-32-16 | 0.0001 | 16 | 0.771 | **0.794** | 0.829 |
| ANN-4 | 128-64-32 | 0.001 | 64 | 0.754 | 0.776 | 0.863 |

→ **ANN-3** đạt F1 cao nhất. Chi tiết phân tích trong
`bao_cao/Chuong_5_Ket_qua_thuc_nghiem.md`.

Biểu đồ kết quả nằm trong thư mục `results/`:
`accuracy_curve.png`, `loss_curve.png`, `confusion_matrix.png`,
`roc_curve.png`, `model_comparison.png`.

---

## 7. Báo cáo

Báo cáo đầy đủ trong thư mục `bao_cao/` (bản Markdown và LaTeX):

| Chương | Nội dung | Người viết |
|---|---|---|
| 1 | Giới thiệu đề tài | Mạnh |
| 2 | Cơ sở lý thuyết (AI, ML, ANN, Backpropagation...) | Mạnh |
| 3 | Dữ liệu và tiền xử lý dữ liệu | Đức |
| 4 | Thiết kế & huấn luyện mô hình ANN | Hùng |
| 5 | Kết quả thực nghiệm | Mạnh |
| 6 | Kết luận và hướng phát triển | Mạnh |
