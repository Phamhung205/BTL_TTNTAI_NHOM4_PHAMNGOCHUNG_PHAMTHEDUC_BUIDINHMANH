"""
data_preprocessing.py
---------------------
Thu thập & TIỀN XỬ LÝ dữ liệu cho bài toán "Đánh giá nguy cơ mắc bệnh tim".
Tác giả: Phạm Thế Đức (Data Engineer & Data Analyst).

Quy trình:
  1. Đọc dữ liệu THÔ từ data/raw/heart.csv (bản tải về từ Kaggle).
  2. Khảo sát: kích thước, kiểu dữ liệu, mô tả thống kê.
  3. Kiểm tra chất lượng: giá trị THIẾU, bản ghi TRÙNG, giá trị NGOẠI LỆ (outlier).
  4. Làm sạch: ép kiểu, điền giá trị thiếu, loại bản ghi trùng, xử lý outlier.
  5. Lưu dữ liệu SẠCH ra data/processed/heart_clean.csv để bước
     feature_engineering.py dùng tiếp.

Chạy:  python src/data_preprocessing.py

GHI CHÚ DỮ LIỆU (bộ heart-disease 14 cột):
  - age      : tuổi (năm)
  - sex      : giới tính (1 = nam, 0 = nữ)
  - cp       : kiểu đau ngực (0-3)
  - trestbps : huyết áp lúc nghỉ (mm Hg)
  - chol     : cholesterol huyết thanh (mg/dl)
  - fbs      : đường huyết lúc đói > 120 mg/dl (1 = đúng, 0 = sai)
  - restecg  : kết quả điện tâm đồ lúc nghỉ (0-2)
  - thalach  : nhịp tim tối đa đạt được
  - exang    : đau thắt ngực khi gắng sức (1 = có, 0 = không)
  - oldpeak  : độ chênh ST khi gắng sức so với lúc nghỉ
  - slope    : độ dốc đoạn ST khi gắng sức (0-2)
  - ca       : số mạch máu lớn nhuộm màu (0-4)
  - thal     : tình trạng thalassemia (0-3)
  - target   : NHÃN -> 1 = có nguy cơ bệnh tim, 0 = không
"""

import os
import pandas as pd

# =========================================================================
# CẤU HÌNH & ĐỊNH NGHĨA CỘT  (dùng chung với feature_engineering.py)
# =========================================================================
RAW_PATH = "data/raw/heart.csv"               # dữ liệu thô đầu vào
CLEAN_PATH = "data/processed/heart_clean.csv"  # dữ liệu sạch đầu ra

TARGET = "target"
# Đặc trưng LIÊN TỤC (số thực) -> sẽ chuẩn hoá ở bước feature engineering
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Đặc trưng NHỊ PHÂN (chỉ nhận 0/1)
BINARY = ["sex", "fbs", "exang"]
# Đặc trưng PHÂN LOẠI rời rạc (đã ở dạng số nguyên)
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]

ALL_FEATURES = CONTINUOUS + BINARY + CATEGORICAL  # 13 đặc trưng


# =========================================================================
# 1. ĐỌC DỮ LIỆU THÔ
# =========================================================================
def load_raw(path=RAW_PATH):
    """Đọc file CSV thô và trả về DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay '{path}'. Hay tai bo du lieu Kaggle "
            f"(johnsmith88/heart-disease-dataset) va luu vao day."
        )
    df = pd.read_csv(path)
    print(f"[1] Da doc du lieu tho: {df.shape[0]} dong x {df.shape[1]} cot")
    return df


# =========================================================================
# 2. KHẢO SÁT NHANH
# =========================================================================
def inspect(df):
    """In thông tin tổng quan: cột, kiểu dữ liệu, mô tả thống kê."""
    print("\n[2] KHAO SAT DU LIEU")
    print("- Cac cot:", list(df.columns))
    print("\n- Kieu du lieu:")
    print(df.dtypes)
    print("\n- 5 dong dau:")
    print(df.head())
    print("\n- Mo ta thong ke (cot so):")
    print(df.describe().T)


# =========================================================================
# 3. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU
# =========================================================================
def check_missing(df):
    """Đếm số ô thiếu của từng cột."""
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print("\n[3a] GIA TRI THIEU (isnull):")
    if miss.empty:
        print("  -> Khong co o thieu.")
    else:
        for col, n in miss.items():
            print(f"  - {col:10s}: {n} o thieu")
    return miss


def check_duplicates(df):
    """Đếm số bản ghi trùng lặp hoàn toàn."""
    n_dup = int(df.duplicated().sum())
    print(f"\n[3b] BAN GHI TRUNG (duplicated): {n_dup} dong")
    return n_dup


def detect_outliers_iqr(df, cols=CONTINUOUS):
    """
    Phát hiện outlier theo quy tắc IQR (Interquartile Range):
      - Q1, Q3 là phân vị 25% và 75%; IQR = Q3 - Q1.
      - Giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR] được coi là outlier.
    Trả về dict {ten_cot: (so_outlier, can_duoi, can_tren)}.
    """
    print("\n[3c] NGOAI LE (outlier - quy tac IQR 1.5):")
    report = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        report[col] = (n_out, low, high)
        print(f"  - {col:10s}: {n_out:3d} outlier "
              f"(nguong [{low:.1f}, {high:.1f}])")
    return report


# =========================================================================
# 4. LÀM SẠCH DỮ LIỆU
# =========================================================================
def clean(df, cap_outliers=True):
    """
    Làm sạch dữ liệu:
      1. Loại bản ghi TRÙNG hoàn toàn.
      2. Điền giá trị THIẾU: cột liên tục -> trung vị (median);
         cột nhị phân/phân loại -> giá trị xuất hiện nhiều nhất (mode).
      3. Ép KIỂU: cột phân loại/nhị phân/nhãn về số nguyên, cột liên tục về float.
      4. (Tuỳ chọn) CẮT outlier về biên IQR (winsorize) để giảm nhiễu.
    """
    print("\n[4] LAM SACH DU LIEU")
    df = df.copy()

    # 4.1 Loại trùng
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  - Da loai {before - len(df)} ban ghi trung "
          f"(con {len(df)} dong).")

    # 4.2 Điền giá trị thiếu
    for col in CONTINUOUS:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"  - Dien thieu '{col}' bang trung vi = {med}")
    for col in BINARY + CATEGORICAL + [TARGET]:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            print(f"  - Dien thieu '{col}' bang mode = {mode}")

    # 4.3 Ép kiểu
    for col in BINARY + CATEGORICAL + [TARGET]:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    for col in CONTINUOUS:
        df[col] = df[col].astype(float)

    # 4.4 Cắt outlier về biên IQR (giữ lại bản ghi, chỉ kéo giá trị cực trị về biên)
    if cap_outliers:
        n_capped = 0
        for col in CONTINUOUS:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_capped += int(((df[col] < low) | (df[col] > high)).sum())
            df[col] = df[col].clip(lower=low, upper=high)
        print(f"  - Da cat {n_capped} gia tri cuc tri ve bien IQR (clip).")

    print(f"  -> Du lieu sach: {df.shape[0]} dong x {df.shape[1]} cot")
    return df


# =========================================================================
# 5. LƯU KẾT QUẢ
# =========================================================================
def save_clean(df, path=CLEAN_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[5] Da luu du lieu sach -> {path}")


# =========================================================================
# HÀM CHÍNH
# =========================================================================
def main():
    df = load_raw()
    inspect(df)

    check_missing(df)
    check_duplicates(df)
    detect_outliers_iqr(df)

    df_clean = clean(df)

    # Kiểm tra lại sau khi làm sạch
    print("\n[KIEM TRA SAU LAM SACH]")
    print("  - Con o thieu :", int(df_clean.isnull().sum().sum()))
    print("  - Con ban trung:", int(df_clean.duplicated().sum()))

    save_clean(df_clean)
    print("\nXONG buoc tien xu ly. Tiep theo chay: python src/feature_engineering.py")


if __name__ == "__main__":
    main()
