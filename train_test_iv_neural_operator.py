from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optional DeepLense Test IV neural-operator baseline on the Common Test I dataset."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/common-test-i"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-path", type=Path, default=Path("models/test_iv_spectral_best.pt"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/test_iv_spectral.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "train_common_test_i.py",
        "--data-root",
        str(args.data_root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--model-type",
        "spectral",
        "--normalize-mode",
        "per_image_standardize",
        "--train-fraction",
        str(args.train_fraction),
        "--val-fraction",
        str(args.val_fraction),
        "--width",
        str(args.width),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--model-path",
        str(args.model_path),
        "--report-path",
        str(args.report_path),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
