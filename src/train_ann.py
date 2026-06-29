"""
train_ann.py
------------
Huấn luyện mô hình ANN bằng scikit-learn (MLPClassifier).

VÌ SAO CÓ FILE NÀY:
  Mô hình "chính thức" của nhóm được thiết kế bằng Keras/TensorFlow
  (src/model.py, src/train.py). Tuy nhiên trên một số máy, TensorFlow bị
  Windows Smart App Control chặn không cho nạp DLL. Để vẫn HUẤN LUYỆN và lấy
  được KẾT QUẢ THẬT phục vụ đánh giá (Chương 5), file này cài đặt một mạng ANN
  TƯƠNG ĐƯƠNG bằng scikit-learn:
      - Kiến trúc: 13 -> 64 -> 32 -> 16 -> 1   (giống model.py)
      - Hàm kích hoạt tầng ẩn: ReLU
      - Bộ tối ưu: Adam,  learning_rate = 0.001,  batch_size = 32
      - Có Early Stopping (theo val_loss) giống bản Keras.

Đầu ra:
  - saved_models/ann_sklearn.joblib : mô hình tốt nhất.
  - results/history.json            : lịch sử train/val (accuracy & loss) để vẽ.

Chạy:  python src/train_ann.py

Tác giả phần đánh giá/thực nghiệm: Bùi Đình Mạnh.
"""

import os
import sys
import json
import copy

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

# =========================================================================
# CẤU HÌNH (đặt giống bản Keras để so sánh công bằng)
# =========================================================================
DATA_PATH = "data/heart.csv"
TARGET_COLUMN = "target"
MODEL_PATH = "saved_models/ann_sklearn.joblib"
HISTORY_PATH = "results/history.json"

HIDDEN_UNITS = (64, 32, 16)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 10            # số epoch chờ trước khi dừng sớm (giống EarlyStopping)
VALIDATION_SPLIT = 0.2   # 20% tập train dùng làm validation


def build_mlp(hidden_units=HIDDEN_UNITS, learning_rate=LEARNING_RATE,
              batch_size=BATCH_SIZE):
    """Khởi tạo MLPClassifier tương đương kiến trúc ANN của nhóm."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_units,
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # L2 regularization nhẹ
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        random_state=42,
    )


def train(data_path=DATA_PATH):
    # ---- 1. Đọc dữ liệu & chia train/test (giống train.py) ----
    X_train, X_test, y_train, y_test = load_data(data_path, target_column=TARGET_COLUMN)

    # ---- 2. Tách thêm validation từ tập train (để theo dõi & dừng sớm) ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SPLIT,
        random_state=42, stratify=y_train,
    )
    print(f"Train: {len(X_tr)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    # ---- 3. Huấn luyện theo từng epoch (partial_fit) để ghi lịch sử ----
    mlp = build_mlp()
    classes = np.array([0, 1])
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    best_val_loss = np.inf
    best_model = None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        mlp.partial_fit(X_tr, y_tr, classes=classes)

        tr_loss = log_loss(y_tr, mlp.predict_proba(X_tr), labels=classes)
        val_loss = log_loss(y_val, mlp.predict_proba(X_val), labels=classes)
        tr_acc = accuracy_score(y_tr, mlp.predict(X_tr))
        val_acc = accuracy_score(y_val, mlp.predict(X_val))

        history["loss"].append(float(tr_loss))
        history["val_loss"].append(float(val_loss))
        history["accuracy"].append(float(tr_acc))
        history["val_accuracy"].append(float(val_acc))

        # Early stopping + giữ lại model tốt nhất theo val_loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model = copy.deepcopy(mlp)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping tai epoch {epoch} "
                      f"(val_loss khong cai thien sau {PATIENCE} epoch).")
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"acc={tr_acc:.4f} val_acc={val_acc:.4f}")

    if best_model is None:
        best_model = mlp

    # ---- 4. Lưu model tốt nhất + lịch sử ----
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\nDa luu model -> {MODEL_PATH}")
    print(f"Da luu lich su ({len(history['loss'])} epoch) -> {HISTORY_PATH}")

    # ---- 5. Báo nhanh kết quả trên tập test ----
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Accuracy tren tap test: {test_acc:.4f}")
    return best_model, history


if __name__ == "__main__":
    train()
