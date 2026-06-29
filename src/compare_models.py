"""
compare_models.py
-----------------
SO SÁNH NHIỀU CẤU HÌNH ANN (thử nghiệm siêu tham số) theo bảng trong đề bài,
huấn luyện trên dữ liệu thật và lập bảng so sánh Accuracy/Precision/Recall/F1/AUC.
Tác giả: Bùi Đình Mạnh (Evaluation Engineer).

4 thí nghiệm (giống bảng phân công của nhóm trưởng):
  | # | Hidden Layers | Learning Rate | Batch Size |
  | 1 | 32-16         | 0.001         | 32         |
  | 2 | 64-32         | 0.001         | 32         |
  | 3 | 64-32-16      | 0.0001        | 16         |
  | 4 | 128-64-32     | 0.001         | 64         |

Đầu ra:
  - results/model_comparison.csv  : bảng số liệu.
  - results/model_comparison.png  : biểu đồ cột so sánh.

Chạy:  python src/compare_models.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

DATA_PATH = "data/heart.csv"
TARGET_COLUMN = "target"
RESULTS_DIR = "results"

# (ten, hidden_layers, learning_rate, batch_size)
EXPERIMENTS = [
    ("ANN-1", (32, 16),       0.001,  32),
    ("ANN-2", (64, 32),       0.001,  32),
    ("ANN-3", (64, 32, 16),   0.0001, 16),
    ("ANN-4", (128, 64, 32),  0.001,  64),
]


def run_experiment(name, hidden, lr, batch, X_train, y_train, X_test, y_test):
    """Huấn luyện 1 cấu hình và trả về dict chỉ số trên tập test."""
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=batch,
        learning_rate_init=lr,
        max_iter=500,
        early_stopping=True,         # tự tách validation & dừng sớm
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Model": name,
        "Hidden Layers": "-".join(map(str, hidden)),
        "LR": lr,
        "Batch": batch,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def plot_comparison(df, out_dir=RESULTS_DIR):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(df))
    width = 0.15
    plt.figure(figsize=(11, 6))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, df[m], width, label=m)
    plt.xticks(x + width * 2, df["Model"])
    plt.ylim(0, 1.05)
    plt.ylabel("Gia tri")
    plt.title("So sanh cac cau hinh ANN tren tap test")
    plt.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    plt.grid(True, axis="y", alpha=0.3)
    path = os.path.join(out_dir, "model_comparison.png")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"Da luu bieu do -> {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(DATA_PATH, target_column=TARGET_COLUMN)

    rows = []
    print("===== CHAY 4 THI NGHIEM SIEU THAM SO =====")
    for name, hidden, lr, batch in EXPERIMENTS:
        print(f"  -> {name}: hidden={hidden}, lr={lr}, batch={batch}")
        rows.append(run_experiment(name, hidden, lr, batch,
                                   X_train, y_train, X_test, y_test))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDa luu bang -> {csv_path}\n")

    # In bảng gọn ra màn hình
    show = df.copy()
    for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        show[m] = show[m].map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))

    best = df.loc[df["F1"].idxmax(), "Model"]
    print(f"\nMo hinh tot nhat theo F1: {best}")

    plot_comparison(df)
    return df


if __name__ == "__main__":
    main()
