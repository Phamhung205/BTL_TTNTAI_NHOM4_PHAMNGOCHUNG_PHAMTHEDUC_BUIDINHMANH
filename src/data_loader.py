"""
data_loader.py
--------------
Đọc dữ liệu đã tiền xử lý từ file CSV và chia thành tập train/test (80/20).

GIẢ ĐỊNH: file CSV đầu vào đã được bạn cùng nhóm xử lý xong:
  - loại bỏ trùng lặp
  - mã hóa biến phân loại (LabelEncoder / OneHotEncoder)
  - chuẩn hóa đặc trưng số (StandardScaler)
File chỉ gồm các cột đặc trưng (đã là số) + 1 cột nhãn 0/1.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path, target_column="target", test_size=0.2, random_state=42):
    """
    Đọc CSV và chia train/test.

    Tham số
    -------
    csv_path : str
        Đường dẫn tới file CSV đã tiền xử lý (ví dụ 'data/heart.csv').
    target_column : str
        Tên cột nhãn (mặc định 'target'). ĐỔI cho khớp dữ liệu thật của nhóm.
    test_size : float
        Tỷ lệ tập test (0.2 = 20%).
    random_state : int
        Cố định để kết quả chia lặp lại được giữa các lần chạy.

    Trả về
    ------
    X_train, X_test, y_train, y_test : numpy.ndarray
    """
    # Đọc dữ liệu
    df = pd.read_csv(csv_path)

    # Kiểm tra cột nhãn có tồn tại không -> báo lỗi rõ ràng nếu sai tên cột
    if target_column not in df.columns:
        raise ValueError(
            f"Khong tim thay cot nhan '{target_column}'. "
            f"Cac cot hien co: {list(df.columns)}"
        )

    # Tách đặc trưng (X) và nhãn (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Chia 80/20. stratify=y giữ tỷ lệ lớp 0/1 cân bằng giữa train và test
    # (quan trọng với dữ liệu y tế thường mất cân bằng).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"So dac trung dau vao (input_dim): {X_train.shape[1]}")
    print(f"Kich thuoc tap train: {X_train.shape[0]} mau")
    print(f"Kich thuoc tap test : {X_test.shape[0]} mau")

    return X_train, X_test, y_train, y_test