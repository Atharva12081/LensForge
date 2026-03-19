from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


CLASSES = ["no", "sphere", "vort"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a radial-feature baseline for Common Test I.")
    parser.add_argument("--data-root", type=Path, default=Path("data/common-test-i"))
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=300)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--report-path", type=Path, default=Path("reports/common_test_i_radial.json"))
    return parser.parse_args()


def radial_features(image: np.ndarray, bins: int) -> np.ndarray:
    h, w = image.shape
    y, x = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = radius / max(radius.max(), 1e-6)

    feats: list[float] = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (radius >= lo) & (radius < hi)
        values = image[mask]
        feats.append(float(values.mean()) if values.size else 0.0)
        feats.append(float(values.std()) if values.size else 0.0)

    center = image[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    feats.extend(
        [
            float(image.mean()),
            float(image.std()),
            float(center.mean()),
            float(center.std()),
            float(image.max()),
        ]
    )
    return np.array(feats, dtype=np.float32)


def load_split(root: Path, split: str, per_class: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASSES):
        files = sorted((root / split / class_name).glob("*.npy"))[:per_class]
        for path in files:
            image = np.load(path)[0].astype(np.float32)
            features.append(radial_features(image, bins=bins))
            labels.append(label)
    return np.stack(features), np.array(labels, dtype=np.int32)


def main() -> None:
    args = parse_args()
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(args.data_root, "train", args.train_per_class, args.bins)
    x_val, y_val = load_split(args.data_root, "val", args.val_per_class, args.bins)

    model = LogisticRegression(max_iter=3000)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(CLASSES), dtype=np.float32)[y_val]

    metrics = {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_roc_auc": float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        ),
    }
    report = {
        "config": {
            "data_root": str(args.data_root),
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "bins": args.bins,
        },
        "validation": metrics,
    }
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
