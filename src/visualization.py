"""
visualization.py
----------------
TRỰC QUAN HÓA kết quả huấn luyện & đánh giá mô hình ANN.
Tác giả: Bùi Đình Mạnh (Evaluation Engineer).

Vẽ và lưu 4 biểu đồ vào thư mục results/:
  - accuracy_curve.png  : độ chính xác train/validation qua các epoch.
  - loss_curve.png      : hàm mất mát train/validation qua các epoch.
  - confusion_matrix.png: ma trận nhầm lẫn trên tập test.
  - roc_curve.png       : đường ROC và chỉ số AUC.

Đầu vào:
  - results/history.json (từ train_ann.py)            -> accuracy & loss curve.
  - model + tập test (qua evaluate.get_predictions)   -> confusion matrix & ROC.

Chạy:  python src/visualization.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # lưu ảnh không cần màn hình
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import get_predictions

RESULTS_DIR = "results"
HISTORY_PATH = "results/history.json"


def plot_accuracy_curve(history, out_dir=RESULTS_DIR):
    plt.figure(figsize=(7, 5))
    plt.plot(history["accuracy"], label="Train")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Accuracy qua cac epoch")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "accuracy_curve.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  -> {path}")


def plot_loss_curve(history, out_dir=RESULTS_DIR):
    plt.figure(figsize=(7, 5))
    plt.plot(history["loss"], label="Train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss qua cac epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss (binary cross-entropy)")
    plt.legend(); plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "loss_curve.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  -> {path}")


def plot_confusion_matrix(y_test, y_pred, out_dir=RESULTS_DIR):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Khong benh (0)", "Co benh (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (tap test)")
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  -> {path}")


def plot_roc_curve(y_test, y_prob, out_dir=RESULTS_DIR):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Ngau nhien")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Duong ROC")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "roc_curve.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  -> {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("===== TRUC QUAN HOA KET QUA =====")

    # 1) Accuracy & Loss curve (từ history.json)
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
        plot_accuracy_curve(history)
        plot_loss_curve(history)
    else:
        print(f"  (Bo qua accuracy/loss curve: chua co {HISTORY_PATH}. "
              f"Hay chay train_ann.py truoc.)")

    # 2) Confusion matrix & ROC (từ model + tập test)
    y_test, y_pred, y_prob = get_predictions()
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    print("Da luu toan bo bieu do vao thu muc results/")


if __name__ == "__main__":
    main()
