from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common_test_i import (
    CommonTestIDataset,
    MultiClassConvNet,
    ResidualClassifier,
    list_multiclass_files,
    maybe_limit_items,
    run_epoch,
    save_report,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Common Test I multiclass baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data/common-test-i"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-type", choices=["conv", "residual"], default="conv")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--input-pool", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--center-crop", type=int, default=0)
    parser.add_argument("--resize-to", type=int, default=0)
    parser.add_argument("--normalize-mode", choices=["none", "per_image_standardize"], default="none")
    parser.add_argument("--disable-augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", type=Path, default=Path("models/common_test_i_best.pt"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/common_test_i_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    train_items = maybe_limit_items(list_multiclass_files(args.data_root, "train"), args.train_fraction, args.seed)
    val_items = maybe_limit_items(list_multiclass_files(args.data_root, "val"), args.val_fraction, args.seed)

    train_loader = DataLoader(
        CommonTestIDataset(
            train_items,
            augment=not args.disable_augment,
            center_crop=args.center_crop if args.center_crop > 0 else None,
            resize_to=args.resize_to if args.resize_to > 0 else None,
            normalize_mode=args.normalize_mode,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CommonTestIDataset(
            val_items,
            augment=False,
            center_crop=args.center_crop if args.center_crop > 0 else None,
            resize_to=args.resize_to if args.resize_to > 0 else None,
            normalize_mode=args.normalize_mode,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model_type == "residual":
        model = ResidualClassifier(width=args.width).to(device)
    else:
        model = MultiClassConvNet(input_pool=args.input_pool, width=args.width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = float("-inf")
    best_val_metrics = None
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train": {
                    "loss": train_metrics.loss,
                    "accuracy": train_metrics.accuracy,
                    "macro_roc_auc": train_metrics.macro_roc_auc,
                },
                "validation": {
                    "loss": val_metrics.loss,
                    "accuracy": val_metrics.accuracy,
                    "macro_roc_auc": val_metrics.macro_roc_auc,
                },
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} train_auc={train_metrics.macro_roc_auc:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} val_auc={val_metrics.macro_roc_auc:.4f}"
        )
        if val_metrics.macro_roc_auc > best_val_auc:
            best_val_auc = val_metrics.macro_roc_auc
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), args.model_path)

    if best_val_metrics is None:
        raise RuntimeError("No validation metrics recorded.")

    save_report(
        report_path=args.report_path,
        config={
            "data_root": str(args.data_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "model_type": args.model_type,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "input_pool": args.input_pool,
            "width": args.width,
            "center_crop": args.center_crop,
            "resize_to": args.resize_to,
            "normalize_mode": args.normalize_mode,
            "disable_augment": args.disable_augment,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": str(device),
            "model_path": str(args.model_path),
        },
        history=history,
        validation_metrics=best_val_metrics,
    )


if __name__ == "__main__":
    main()
