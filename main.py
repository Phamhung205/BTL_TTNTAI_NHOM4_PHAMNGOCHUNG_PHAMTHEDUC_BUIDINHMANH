"""
main.py
-------
Chương trình CHÍNH (giao diện Console) điều phối toàn bộ quy trình của dự án
"Đánh giá nguy cơ mắc bệnh tim bằng ANN".

Pipeline gồm 6 bước:
  1. Tiền xử lý dữ liệu      (src/data_preprocessing.py)
  2. Feature engineering     (src/feature_engineering.py)
  3. Huấn luyện mô hình ANN  (src/train_ann.py)
  4. Đánh giá mô hình        (src/evaluate.py)
  5. Trực quan hóa kết quả   (src/visualization.py)
  6. So sánh các cấu hình    (src/compare_models.py)

Cách dùng:
  python main.py            -> hiện menu để chọn từng bước
  python main.py all        -> chạy tự động toàn bộ pipeline (1 -> 6)
  python main.py 1 2 3      -> chạy các bước được chỉ định

Ghi chú: bước 3 dùng scikit-learn (MLPClassifier) làm bản ANN chạy được trên mọi
máy. Nếu môi trường có TensorFlow, có thể chạy bản Keras gốc: python src/train.py
"""

import os
import sys

# Cho phép import các module trong thư mục src/
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def step_preprocess():
    import data_preprocessing
    data_preprocessing.main()


def step_feature_engineering():
    import feature_engineering
    feature_engineering.main()


def step_train():
    import train_ann
    train_ann.train()


def step_evaluate():
    import evaluate
    evaluate.main()


def step_visualize():
    import visualization
    visualization.main()


def step_compare():
    import compare_models
    compare_models.main()


STEPS = {
    "1": ("Tien xu ly du lieu", step_preprocess),
    "2": ("Feature engineering (ma hoa, chuan hoa, chia train/test)", step_feature_engineering),
    "3": ("Huan luyen mo hinh ANN", step_train),
    "4": ("Danh gia mo hinh (Accuracy, Precision, Recall, F1, AUC)", step_evaluate),
    "5": ("Truc quan hoa ket qua (cac bieu do)", step_visualize),
    "6": ("So sanh cac cau hinh ANN", step_compare),
}


def run_steps(keys):
    for k in keys:
        name, func = STEPS[k]
        print("\n" + "=" * 60)
        print(f"BUOC {k}: {name}")
        print("=" * 60)
        func()


def run_all():
    run_steps(["1", "2", "3", "4", "5", "6"])
    print("\n>>> DA HOAN THANH TOAN BO PIPELINE. Ket qua trong thu muc results/ <<<")


def menu():
    while True:
        print("\n" + "=" * 60)
        print(" DU AN: DANH GIA NGUY CO MAC BENH TIM BANG ANN")
        print("=" * 60)
        for k, (name, _) in STEPS.items():
            print(f"  {k}. {name}")
        print("  0. Chay TOAN BO pipeline (1 -> 6)")
        print("  q. Thoat")
        choice = input("Chon: ").strip().lower()

        if choice == "q":
            print("Tam biet!")
            break
        elif choice == "0":
            run_all()
        elif choice in STEPS:
            run_steps([choice])
        else:
            print("Lua chon khong hop le, thu lai.")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        menu()
    elif "all" in args:
        run_all()
    else:
        valid = [a for a in args if a in STEPS]
        if valid:
            run_steps(valid)
        else:
            print("Tham so khong hop le. Dung: python main.py [all | 1 2 3 ...]")
