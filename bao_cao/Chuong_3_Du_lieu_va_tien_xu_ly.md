
# CHƯƠNG 3. DỮ LIỆU VÀ TIỀN XỬ LÝ DỮ LIỆU

> Người thực hiện: **Phạm Thế Đức** — Data Engineer & Data Analyst
> Mã nguồn liên quan: `src/data_preprocessing.py`, `src/feature_engineering.py`, `notebook/data_exploration.ipynb`
> Biểu đồ minh hoạ: thư mục `bao_cao/figures/`

---

## 3.1. Giới thiệu bộ dữ liệu

Bộ dữ liệu sử dụng là **Heart Disease Dataset** trên Kaggle
(`johnsmith88/heart-disease-dataset`), tổng hợp từ bốn cơ sở dữ liệu y tế
(Cleveland, Hungary, Switzerland, Long Beach V) từ năm 1988. Tuy bản gốc có 76
thuộc tính, các nghiên cứu công bố đều dùng **tập con 14 thuộc tính** — đây cũng
là bộ được dùng trong đề tài.

- **Số bản ghi ban đầu:** 1025 dòng.
- **Số thuộc tính:** 13 đặc trưng đầu vào + 1 biến mục tiêu `target`.
- **Bài toán:** phân loại nhị phân — dự đoán bệnh nhân **có nguy cơ mắc bệnh tim
  (`target = 1`)** hay **không (`target = 0`)**.

**Bảng 3.1 — Mô tả các thuộc tính**

| # | Thuộc tính | Ý nghĩa | Kiểu |
|---|---|---|---|
| 1 | `age` | Tuổi (năm) | Liên tục |
| 2 | `sex` | Giới tính (1 = nam, 0 = nữ) | Nhị phân |
| 3 | `cp` | Kiểu đau ngực (0–3) | Phân loại |
| 4 | `trestbps` | Huyết áp lúc nghỉ (mm Hg) | Liên tục |
| 5 | `chol` | Cholesterol huyết thanh (mg/dl) | Liên tục |
| 6 | `fbs` | Đường huyết lúc đói > 120 mg/dl (1/0) | Nhị phân |
| 7 | `restecg` | Kết quả điện tâm đồ lúc nghỉ (0–2) | Phân loại |
| 8 | `thalach` | Nhịp tim tối đa đạt được | Liên tục |
| 9 | `exang` | Đau thắt ngực khi gắng sức (1/0) | Nhị phân |
| 10 | `oldpeak` | Độ chênh đoạn ST khi gắng sức so với nghỉ | Liên tục |
| 11 | `slope` | Độ dốc đoạn ST khi gắng sức (0–2) | Phân loại |
| 12 | `ca` | Số mạch máu lớn nhuộm màu (0–4) | Phân loại |
| 13 | `thal` | Tình trạng thalassemia (0–3) | Phân loại |
| 14 | **`target`** | **Nhãn: 1 = có nguy cơ, 0 = không** | Nhị phân |

> Ghi chú: trong kho dữ liệu còn có `heart-selected-columns.csv` (chỉ gồm 10 cột,
> **không có** cột `target`). File này không dùng để huấn luyện được, nên đề tài
> sử dụng file đầy đủ `heart.csv`.

---

## 3.2. Thống kê mô tả

Sau khi nạp dữ liệu, các đặc trưng số được mô tả thống kê (giá trị nhỏ nhất, lớn
nhất, trung bình, độ lệch chuẩn, các phân vị). Một số quan sát chính (trên 302
bản ghi duy nhất — xem mục 3.3):

| Đặc trưng | Min | Max | Trung bình | Nhận xét |
|---|---|---|---|---|
| `age` | 29 | 77 | 54.4 | Bệnh nhân chủ yếu trung niên – cao tuổi |
| `trestbps` | 94 | 200 | 131.6 | Huyết áp dao động rộng |
| `chol` | 126 | 564 | 246.5 | Có giá trị rất cao (564) → cần xét outlier |
| `thalach` | 71 | 202 | 149.6 | Nhịp tim tối đa |
| `oldpeak` | 0 | 6.2 | 1.04 | Lệch phải, nhiều giá trị 0 |

**Nhận xét:** các đặc trưng liên tục có **thang đo rất khác nhau** (vd `chol` cỡ
hàng trăm trong khi `oldpeak` chỉ 0–6) → bắt buộc phải **chuẩn hoá** trước khi
đưa vào mạng nơ-ron (mục 3.5).

---

## 3.3. Kiểm tra chất lượng dữ liệu

### a) Giá trị thiếu
Kiểm tra bằng `df.isnull().sum()`: **không có ô thiếu** trong toàn bộ dữ liệu.

### b) Bản ghi trùng lặp — vấn đề quan trọng nhất
Kiểm tra bằng `df.duplicated().sum()`:

> **1025 bản ghi nhưng có 723 bản ghi TRÙNG LẶP hoàn toàn → chỉ còn 302 bản ghi duy nhất.**

Đây là đặc điểm đã được cộng đồng ghi nhận của phiên bản dữ liệu này (được nhân
bản từ bộ Cleveland gốc 303 dòng). Nếu **giữ nguyên** các bản trùng, một bệnh
nhân có thể xuất hiện đồng thời ở **cả tập train lẫn test** → gây **rò rỉ dữ liệu
(data leakage)**, khiến độ chính xác báo cáo bị "ảo" (cao giả tạo). Vì vậy nhóm
**loại bỏ toàn bộ bản trùng** trước khi chia dữ liệu và huấn luyện.

### c) Giá trị ngoại lệ (Outlier)
Dùng quy tắc **IQR** (khoảng tứ phân vị): giá trị nằm ngoài
`[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` được coi là ngoại lệ. Kết quả phát hiện một số
outlier ở `trestbps`, `chol`, `thalach`, `oldpeak`. Cách xử lý: **cắt về biên
(winsorize/clip)** — giữ lại bản ghi nhưng kéo giá trị cực trị về ngưỡng IQR, để
giảm nhiễu mà không làm mất mẫu (quan trọng khi dữ liệu chỉ còn 302 dòng).

### d) Giá trị bất thường về mặt mã hoá
- `ca` xuất hiện giá trị **4** (mô tả gốc chỉ 0–3).
- `thal` xuất hiện giá trị **0** (mô tả gốc chỉ 1–3).

Đây là các giá trị nằm ngoài bảng mã chuẩn. Trong phạm vi đề tài, chúng được giữ
lại như **một mức phân loại riêng** (không loại bỏ) vì số lượng nhỏ và mô hình
ANN vẫn xử lý được dưới dạng số.

---

## 3.4. Tiền xử lý dữ liệu — `src/data_preprocessing.py`

Quy trình làm sạch được cài đặt thành các bước rõ ràng:

1. **Đọc dữ liệu thô** từ `data/raw/heart.csv`.
2. **Loại bản ghi trùng** bằng `drop_duplicates()` → 1025 → **302 dòng**.
3. **Điền giá trị thiếu** (phòng trường hợp dữ liệu thật khác có thiếu):
   - Đặc trưng liên tục → điền bằng **trung vị (median)**.
   - Đặc trưng phân loại/nhị phân → điền bằng **giá trị phổ biến nhất (mode)**.
4. **Chuẩn hoá kiểu dữ liệu:** ép các cột phân loại/nhị phân/nhãn về `int`, các
   cột liên tục về `float`.
5. **Xử lý outlier:** cắt giá trị cực trị về biên IQR.
6. **Lưu** dữ liệu sạch ra `data/processed/heart_clean.csv` (302 dòng × 14 cột).

Kết quả kiểm tra lại sau khi làm sạch: **0 ô thiếu, 0 bản trùng.**

---

## 3.5. Feature Engineering — `src/feature_engineering.py`

1. **Mã hoá biến phân loại:** dữ liệu vốn đã ở dạng số nguyên nên không cần mã
   hoá thêm; tuy nhiên pipeline vẫn áp dụng `LabelEncoder` cho mọi cột dạng chữ
   (nếu có) để bảo đảm tính tổng quát. Giữ nguyên **13 đặc trưng** để khớp với
   lớp đầu vào `Input(13)` của mô hình ANN.
2. **Chuẩn hoá đặc trưng liên tục** (`age, trestbps, chol, thalach, oldpeak`)
   bằng **`StandardScaler`** (đưa về trung bình 0, độ lệch chuẩn 1). Các cột nhị
   phân/phân loại giữ nguyên giá trị.
3. **Chống rò rỉ dữ liệu:** `StandardScaler` được **fit chỉ trên tập train**, sau
   đó áp dụng (`transform`) cho cả train và test. Scaler được lưu lại ở
   `saved_models/scaler.pkl` để tái sử dụng khi dự đoán dữ liệu mới.

---

## 3.6. Chia tập dữ liệu train/test

Sử dụng `train_test_split()` với tỷ lệ **80% train – 20% test**,
`stratify=target` để **giữ cân bằng tỷ lệ nhãn** giữa hai tập, `random_state=42`
để kết quả lặp lại được.

| Tập | Số mẫu | Nhãn 1 (có bệnh) | Nhãn 0 (không bệnh) |
|---|---|---|---|
| Train | 241 | 131 | 110 |
| Test | 61 | 33 | 28 |
| **Tổng** | **302** | **164** | **138** |

Kết quả được lưu thành hai file đúng theo phân công:
`data/processed/train.csv` và `data/processed/test.csv`.

> **Lưu ý phối hợp với nhóm trưởng:** file `src/train.py` hiện đọc một file tổng
> rồi tự chia 80/20. Để tương thích, `feature_engineering.py` đồng thời xuất thêm
> file tổng đã xử lý `data/heart.csv` (Phương án A) — nhóm trưởng có thể chạy
> `train.py` ngay mà **không phải sửa code**. Nếu muốn dùng trực tiếp hai file
> train/test (Phương án B), chỉ cần chỉnh `data_loader.py` đọc 2 file.

---

## 3.7. Phân tích dữ liệu khám phá (EDA) — `notebook/data_exploration.ipynb`

### a) Phân bố biến mục tiêu
Hai lớp khá **cân bằng** (≈ 54% có bệnh, 46% không) → thuận lợi cho huấn luyện,
ít cần kỹ thuật xử lý mất cân bằng.
*(Hình: `figures/target_distribution.png`)*

### b) Phân bố độ tuổi
Bệnh nhân tập trung ở độ tuổi 40–65; phân bố theo nhóm bệnh có sự khác biệt.
*(Hình: `figures/age_distribution.png`)*

### c) Giới tính
Tỷ lệ nam cao hơn nữ (≈ 68% nam). Có sự khác biệt rõ về nguy cơ giữa hai giới.
*(Hình: `figures/sex_vs_target.png`)*

### d) Đặc trưng liên tục (Histogram & Boxplot)
`chol` và `oldpeak` lệch phải, có outlier; `thalach` gần phân phối chuẩn.
Boxplot theo nhóm bệnh cho thấy nhóm có bệnh thường có `thalach` cao hơn và
`oldpeak` thấp hơn.
*(Hình: `figures/continuous_histograms.png`, `figures/continuous_boxplots.png`,
`figures/continuous_by_target.png`)*

### e) Biến phân loại (Countplot)
`cp` (kiểu đau ngực), `exang`, `ca`, `thal` phân bố khác biệt rõ giữa hai nhóm
bệnh → là các đặc trưng có giá trị phân loại tốt.
*(Hình: `figures/categorical_countplots.png`)*

### f) Ma trận tương quan (Correlation Heatmap)
*(Hình: `figures/correlation_heatmap.png`)*

**Bảng 3.2 — Tương quan của từng thuộc tính với `target`** (sắp theo độ lớn):

| Thuộc tính | Hệ số tương quan | | Thuộc tính | Hệ số tương quan |
|---|---|---|---|---|
| `exang` | −0.436 | | `thal` | −0.343 |
| `cp` | +0.432 | | `sex` | −0.284 |
| `oldpeak` | −0.429 | | `age` | −0.221 |
| `thalach` | +0.420 | | `trestbps` | −0.146 |
| `ca` | −0.409 | | `restecg` | +0.135 |
| `slope` | +0.344 | | `chol` | −0.081 |
| | | | `fbs` | −0.027 |

**Nhận xét:** các thuộc tính tương quan mạnh nhất với nguy cơ bệnh tim là
`exang`, `cp`, `oldpeak`, `thalach`, `ca` — phù hợp với y văn (đau ngực, đau khi
gắng sức, biến đổi đoạn ST và số mạch tổn thương là các chỉ dấu tim mạch quan
trọng). Hai thuộc tính `chol` và `fbs` tương quan yếu nhất với nhãn.

---

## 3.8. Kết luận chương

Chương 3 đã hoàn thành toàn bộ khâu chuẩn bị dữ liệu cho mô hình ANN:

- Làm sạch dữ liệu: phát hiện và **loại 723 bản ghi trùng** (còn 302 dòng), xác
  nhận không có giá trị thiếu, xử lý outlier bằng IQR.
- Feature engineering: chuẩn hoá đặc trưng liên tục bằng `StandardScaler`
  (fit trên train, tránh rò rỉ dữ liệu), giữ nguyên 13 đặc trưng.
- Chia dữ liệu **80/20 có phân tầng**, xuất `train.csv` / `test.csv`.
- EDA cho thấy dữ liệu cân bằng nhãn và xác định được nhóm đặc trưng quan trọng
  (`exang`, `cp`, `oldpeak`, `thalach`, `ca`).

Bộ dữ liệu đã sẵn sàng để chuyển sang **Chương 4 — Thiết kế và huấn luyện mô hình
ANN**.
