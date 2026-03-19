from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


CLASSES = ["no", "sphere", "vort"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HOG-feature baseline for Common Test I.")
    parser.add_argument("--data-root", type=Path, default=Path("data/common-test-i"))
    parser.add_argument("--train-per-class", type=int, default=800)
    parser.add_argument("--val-per-class", type=int, default=250)
    parser.add_argument("--report-path", type=Path, default=Path("reports/common_test_i_hog.json"))
    return parser.parse_args()


def extract_hog(image: np.ndarray) -> np.ndarray:
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def load_split(root: Path, split: str, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASSES):
        files = sorted((root / split / class_name).glob("*.npy"))[:per_class]
        for path in files:
            image = np.load(path)[0].astype(np.float32)
            features.append(extract_hog(image))
            labels.append(label)
    return np.stack(features), np.array(labels, dtype=np.int32)


def main() -> None:
    args = parse_args()
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(args.data_root, "train", args.train_per_class)
    x_val, y_val = load_split(args.data_root, "val", args.val_per_class)

    model = LogisticRegression(max_iter=3000)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(CLASSES), dtype=np.float32)[y_val]

    report = {
        "config": {
            "data_root": str(args.data_root),
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "feature_type": "hog",
        },
        "validation": {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "macro_roc_auc": float(
                roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
            ),
        },
    }
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
