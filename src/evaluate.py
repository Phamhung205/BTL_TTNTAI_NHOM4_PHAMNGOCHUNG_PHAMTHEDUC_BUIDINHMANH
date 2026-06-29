"""
evaluate.py
-----------
ĐÁNH GIÁ mô hình ANN trên tập test cho bài toán đánh giá nguy cơ bệnh tim.
Tác giả: Bùi Đình Mạnh (Evaluation Engineer).

Các chỉ số tính toán (dùng sklearn.metrics):
  - Accuracy   : tỷ lệ dự đoán đúng tổng thể.
  - Precision  : trong số ca bị dự đoán "có bệnh", bao nhiêu % thực sự có bệnh.
  - Recall     : trong số ca thực sự có bệnh, mô hình bắt được bao nhiêu %.
  - F1-Score   : trung bình điều hòa của Precision và Recall.
  - ROC-AUC    : khả năng phân tách hai lớp (0.5 = ngẫu nhiên, 1.0 = hoàn hảo).

Đầu vào:
  - Model đã huấn luyện: saved_models/ann_sklearn.joblib (chạy train_ann.py trước).
  - Dữ liệu: data/heart.csv (chia train/test GIỐNG lúc huấn luyện -> đúng tập test).

Đầu ra:
  - In bảng chỉ số + confusion matrix + classification_report ra màn hình.
  - Lưu results/metrics.json và results/metrics.txt.

Chạy:  python src/evaluate.py
"""

import os
import sys
import json

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

# =========================================================================
# CẤU HÌNH
# =========================================================================
DATA_PATH = "data/heart.csv"
TARGET_COLUMN = "target"
MODEL_PATH = "saved_models/ann_sklearn.joblib"
RESULTS_DIR = "results"


def get_predictions(model_path=MODEL_PATH, data_path=DATA_PATH):
    """
    Nạp model + dữ liệu test, trả về (y_test, y_pred, y_prob).
    Dùng cùng cách chia dữ liệu với lúc huấn luyện để đúng tập test.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Khong tim thay model '{model_path}'. Hay chay truoc: python src/train_ann.py"
        )
    X_train, X_test, y_train, y_test = load_data(data_path, target_column=TARGET_COLUMN)
    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_test)[:, 1]   # xác suất thuộc lớp 1 (có bệnh)
    y_pred = (y_prob >= 0.5).astype(int)         # ngưỡng 0.5 -> nhãn 0/1
    return y_test, y_pred, y_prob


def compute_metrics(y_test, y_pred, y_prob):
    """Tính toàn bộ chỉ số đánh giá, trả về dict."""
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def save_metrics(metrics, y_test, y_pred):
    """Lưu chỉ số ra results/metrics.json và results/metrics.txt."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(RESULTS_DIR, "metrics.txt")
    cm = confusion_matrix(y_test, y_pred)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("KET QUA DANH GIA MO HINH ANN TREN TAP TEST\n")
        f.write("=" * 45 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n")
        f.write("\nConfusion matrix [[TN, FP], [FN, TP]]:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_pred,
                                      target_names=["Khong benh (0)", "Co benh (1)"],
                                      zero_division=0))
    print(f"Da luu chi so -> {json_path} | {txt_path}")


def main():
    print("===== DANH GIA MO HINH ANN =====")
    y_test, y_pred, y_prob = get_predictions()
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("\n--- Cac chi so tren tap test ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n--- Confusion matrix [[TN, FP], [FN, TP]] ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification report ---")
    print(classification_report(y_test, y_pred,
                                target_names=["Khong benh (0)", "Co benh (1)"],
                                zero_division=0))

    save_metrics(metrics, y_test, y_pred)
    return metrics


if __name__ == "__main__":
    main()
