"""
model.py
--------
Thiết kế kiến trúc mạng nơ-ron nhân tạo (ANN) cho bài toán
phân loại nhị phân: Đánh giá nguy cơ mắc bệnh tim.

Kiến trúc: Input -> Hidden 1 -> Hidden 2 -> Hidden 3 -> Output
Tác giả: Nhóm trưởng - phụ trách thiết kế & huấn luyện mô hình.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim,
                hidden_units=(64, 32, 16),
                dropout_rate=0.0,
                learning_rate=0.001):
    """
    Xây dựng mô hình ANN cho phân loại nhị phân nguy cơ bệnh tim.

    Tham số
    -------
    input_dim : int
        Số đặc trưng đầu vào (số cột của X sau khi tiền xử lý).
    hidden_units : tuple
        Số nơ-ron của từng tầng ẩn. Mặc định (64, 32, 16) -> 3 tầng ẩn.
        Truyền tuple khác để thử nghiệm kiến trúc khác, ví dụ (32, 16).
    dropout_rate : float
        Tỷ lệ Dropout sau mỗi tầng ẩn (0.0 = không dùng). Giúp giảm overfitting.
    learning_rate : float
        Tốc độ học cho bộ tối ưu Adam.

    Trả về
    ------
    model : keras.Model
        Mô hình đã compile, sẵn sàng để huấn luyện.
    """

    # ---- Khởi tạo mô hình tuần tự (Sequential) ----
    model = keras.Sequential(name="Heart_Disease_ANN")

    # ---- INPUT LAYER ----
    # Lớp Input khai báo hình dạng dữ liệu đầu vào: mỗi mẫu là một vector
    # có độ dài = input_dim (số đặc trưng). Không có tham số học ở lớp này.
    model.add(keras.Input(shape=(input_dim,), name="input_layer"))

    # ---- HIDDEN LAYERS ----
    # Lần lượt thêm các tầng ẩn Dense (kết nối đầy đủ).
    # - ReLU: hàm kích hoạt phi tuyến, giúp mạng học quan hệ phức tạp,
    #   đồng thời giảm hiện tượng vanishing gradient so với sigmoid/tanh.
    # - Số nơ-ron giảm dần (64 -> 32 -> 16): mạng nén dần thông tin,
    #   trích xuất đặc trưng từ tổng quát đến cô đọng trước khi ra quyết định.
    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units,
                               activation="relu",
                               name=f"hidden_layer_{i}"))
        # Dropout (tùy chọn): ngẫu nhiên "tắt" một phần nơ-ron khi huấn luyện
        # để mô hình không phụ thuộc quá mức vào vài nơ-ron -> chống overfitting.
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate,
                                     name=f"dropout_{i}"))

    # ---- OUTPUT LAYER ----
    # 1 nơ-ron + sigmoid: cho ra xác suất thuộc lớp "có nguy cơ" trong [0, 1].
    # Ngưỡng 0.5 thường dùng để quy về nhãn 0/1.
    model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))

    # ---- COMPILE ----
    # - Loss: BinaryCrossentropy — chuẩn cho phân loại nhị phân.
    # - Optimizer: Adam — hội tụ nhanh, ổn định, ít phải tinh chỉnh thủ công.
    # - Metrics: theo dõi accuracy (và có thể thêm AUC, Precision, Recall).
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    return model


# Chạy trực tiếp file để kiểm tra nhanh kiến trúc và in summary.
if __name__ == "__main__":
    # Giả định 13 đặc trưng (ví dụ bộ dữ liệu UCI Heart Disease).
    demo = build_model(input_dim=13)
    demo.summary()