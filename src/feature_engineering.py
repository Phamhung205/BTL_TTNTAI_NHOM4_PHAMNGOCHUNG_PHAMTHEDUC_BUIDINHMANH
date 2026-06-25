"""
feature_engineering.py
----------------------
FEATURE ENGINEERING cho bài toán "Đánh giá nguy cơ mắc bệnh tim".
Tác giả: Phạm Thế Đức (Data Engineer & Data Analyst).

Đầu vào : data/processed/heart_clean.csv  (từ data_preprocessing.py)
Quy trình:
  1. Mã hoá biến phân loại (LabelEncoder) -> đảm bảo mọi cột đều là số.
  2. Chia tập train/test theo tỷ lệ 80/20 (stratify giữ cân bằng nhãn 0/1).
  3. Chuẩn hoá đặc trưng liên tục bằng StandardScaler
     (FIT trên tập TRAIN, rồi transform cả train lẫn test -> tránh rò rỉ dữ liệu).
  4. Xuất kết quả phục vụ bước huấn luyện:
       - data/processed/train.csv   (Phương án B - đã tách sẵn)
       - data/processed/test.csv    (Phương án B)
       - data/heart.csv             (Phương án A - 1 file tổng để train.py tự chia)
       - saved_models/scaler.pkl    (lưu scaler để tái sử dụng khi dự đoán)

Chạy:  python src/feature_engineering.py

GIỮ NGUYÊN 13 đặc trưng + cột nhãn 'target' để khớp với model của nhóm trưởng
(Input(13) -> Total params 3,521). Tham số chia (test_size=0.2, random_state=42,
stratify) đặt GIỐNG src/data_loader.py để hai bên nhất quán.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

try:
    import joblib  # đi kèm scikit-learn
except ImportError:  # phòng trường hợp hiếm
    joblib = None

# Dùng lại định nghĩa cột từ bước tiền xử lý (một nguồn sự thật duy nhất)
from data_preprocessing import (
    CLEAN_PATH, TARGET, CONTINUOUS, BINARY, CATEGORICAL, ALL_FEATURES,
)

# =========================================================================
# CẤU HÌNH
# =========================================================================
TRAIN_PATH = "data/processed/train.csv"   # Phương án B
TEST_PATH = "data/processed/test.csv"     # Phương án B
COMBINED_PATH = "data/heart.csv"          # Phương án A (train.py đọc file này)
SCALER_PATH = "saved_models/scaler.pkl"

TEST_SIZE = 0.2        # khớp data_loader.py
RANDOM_STATE = 42      # khớp data_loader.py


# =========================================================================
# 1. ĐỌC DỮ LIỆU SẠCH
# =========================================================================
def load_clean(path=CLEAN_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay '{path}'. Hay chay truoc: "
            f"python src/data_preprocessing.py"
        )
    df = pd.read_csv(path)
    print(f"[1] Da doc du lieu sach: {df.shape[0]} dong x {df.shape[1]} cot")
    return df


# =========================================================================
# 2. MÃ HOÁ BIẾN PHÂN LOẠI
# =========================================================================
def encode(df):
    """
    Mã hoá mọi cột chưa phải dạng số bằng LabelEncoder.
    Bộ heart-disease vốn đã ở dạng số nguyên, nhưng bước này giúp pipeline
    vẫn chạy đúng nếu dữ liệu thật chứa nhãn dạng chữ (vd 'male'/'female').
    """
    df = df.copy()
    encoded_cols = []
    for col in BINARY + CATEGORICAL:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            encoded_cols.append(col)
    if encoded_cols:
        print(f"[2] Da ma hoa (LabelEncoder) cac cot: {encoded_cols}")
    else:
        print("[2] Du lieu da o dang so - khong can ma hoa them.")
    return df


# =========================================================================
# 3. CHIA TRAIN/TEST + 4. CHUẨN HOÁ
# =========================================================================
def split_and_scale(df):
    """
    Chia 80/20 (stratify theo nhãn) rồi chuẩn hoá đặc trưng liên tục.
    StandardScaler được FIT chỉ trên TRAIN để không làm rò rỉ thông tin test.
    Trả về (train_df, test_df, scaler).
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[3] Chia du lieu: train={len(X_train)} | test={len(X_test)} "
          f"(ty le test={TEST_SIZE:.0%}, stratify theo nhan)")

    # Chuẩn hoá CHỈ các cột liên tục; cột nhị phân/phân loại giữ nguyên giá trị
    scaler = StandardScaler()
    X_train[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])
    X_test[CONTINUOUS] = scaler.transform(X_test[CONTINUOUS])
    print(f"[4] Da chuan hoa (StandardScaler, fit tren train) cac cot: {CONTINUOUS}")

    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    return train_df, test_df, scaler


# =========================================================================
# 5. LƯU KẾT QUẢ
# =========================================================================
def save_outputs(train_df, test_df, scaler):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # --- Phương án B: 2 file tách sẵn ---
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"\n[5] (Phuong an B) Da luu -> {TRAIN_PATH} | {TEST_PATH}")

    # --- Phương án A: 1 file tổng (train.py cua nhom truong se tu chia 80/20) ---
    # Gộp lại train+test đã xử lý thành 1 file tổng, giữ thứ tự cột chuẩn.
    combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    combined = combined[ALL_FEATURES + [TARGET]]
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"    (Phuong an A) Da luu file tong -> {COMBINED_PATH}")

    # --- Lưu scaler để dùng lại khi dự đoán dữ liệu mới ---
    if joblib is not None:
        joblib.dump(scaler, SCALER_PATH)
        print(f"    Da luu scaler -> {SCALER_PATH}")


# =========================================================================
# HÀM CHÍNH
# =========================================================================
def main():
    df = load_clean()
    df = encode(df)
    train_df, test_df, scaler = split_and_scale(df)
    save_outputs(train_df, test_df, scaler)

    print("\nTOM TAT:")
    print(f"  - So dac trung dau vao (input_dim): {len(ALL_FEATURES)}")
    print(f"  - Phan bo nhan tap train:\n{train_df[TARGET].value_counts().to_string()}")
    print(f"  - Phan bo nhan tap test :\n{test_df[TARGET].value_counts().to_string()}")
    print("\nXONG feature engineering. Nhom truong co the chay: python src/train.py")


if __name__ == "__main__":
    import sys
    # Cho phép chạy bằng "python src/feature_engineering.py" (import cùng thư mục)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
