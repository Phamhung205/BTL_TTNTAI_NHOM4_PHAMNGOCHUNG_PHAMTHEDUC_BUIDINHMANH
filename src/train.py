"""
train.py
--------
Huấn luyện mô hình ANN cho bài toán đánh giá nguy cơ mắc bệnh tim.

Quy trình:
  1. Đọc & chia dữ liệu (80/20) qua data_loader.
  2. Dựng mô hình qua build_model() trong model.py.
  3. Huấn luyện với Adam + BinaryCrossentropy.
  4. Dùng EarlyStopping (chống overfitting) + ModelCheckpoint (lưu model tốt nhất).
  5. Lưu best_model.h5 và ann_model.h5 vào saved_models/.
  6. Vẽ biểu đồ accuracy/loss để đưa vào báo cáo.

Chạy:  python src/train.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # vẽ và lưu ảnh không cần màn hình (an toàn khi chạy server)
import matplotlib.pyplot as plt

from tensorflow import keras

# Import 2 module cùng thư mục src/
# Khi chạy bằng "python src/train.py", thêm thư mục hiện tại vào đường dẫn import.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data_loader import load_data


# =========================================================================
# CẤU HÌNH — chỉnh các giá trị ở đây
# =========================================================================
DATA_PATH = "data/heart.csv"      # đường dẫn dữ liệu (đổi nếu cần)
TARGET_COLUMN = "target"          # tên cột nhãn (đổi cho khớp dữ liệu nhóm)
SAVED_DIR = "saved_models"        # thư mục lưu model

# Siêu tham số huấn luyện (giá trị gợi ý hợp lý cho bài toán này)
LEARNING_RATE = 0.001             # tốc độ học cho Adam
BATCH_SIZE = 32                   # số mẫu mỗi lần cập nhật trọng số
EPOCHS = 100                      # số vòng tối đa (EarlyStopping sẽ dừng sớm)
HIDDEN_UNITS = (64, 32, 16)       # kiến trúc 3 tầng ẩn
VALIDATION_SPLIT = 0.2            # 20% tập train dùng để validation


def main():
    # ---- 1. Đọc & chia dữ liệu ----
    X_train, X_test, y_train, y_test = load_data(
        DATA_PATH, target_column=TARGET_COLUMN
    )
    input_dim = X_train.shape[1]

    # ---- 2. Dựng mô hình ----
    model = build_model(
        input_dim=input_dim,
        hidden_units=HIDDEN_UNITS,
        learning_rate=LEARNING_RATE,
    )
    model.summary()

    # ---- 3. Chuẩn bị thư mục lưu model ----
    os.makedirs(SAVED_DIR, exist_ok=True)
    best_path = os.path.join(SAVED_DIR, "best_model.h5")
    final_path = os.path.join(SAVED_DIR, "ann_model.h5")

    # ---- 4. Callback ----
    # EarlyStopping: theo dõi val_loss; nếu sau 'patience' epoch không cải thiện
    # thì dừng, và khôi phục trọng số tốt nhất -> tránh overfitting.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    # ModelCheckpoint: tự lưu lại model có val_loss thấp nhất ra best_model.h5.
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=best_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # ---- 5. Huấn luyện ----
    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    # ---- 6. Lưu model cuối cùng (epoch cuối) ----
    model.save(final_path)
    print(f"\nDa luu model tot nhat  -> {best_path}")
    print(f"Da luu model cuoi cung -> {final_path}")

    # ---- 7. Đánh giá trên tập test ----
    print("\n===== Danh gia tren tap test =====")
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(f"{name:12s}: {value:.4f}")

    # ---- 8. Vẽ biểu đồ accuracy & loss (phục vụ báo cáo Chương 5) ----
    plot_history(history)


def plot_history(history):
    """Vẽ đường accuracy và loss theo epoch, lưu ra file PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Biểu đồ Accuracy
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy qua cac epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Biểu đồ Loss
    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss qua cac epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVED_DIR, "training_history.png")
    plt.savefig(out_path, dpi=120)
    print(f"Da luu bieu do qua trinh huan luyen -> {out_path}")


if __name__ == "__main__":
    main()