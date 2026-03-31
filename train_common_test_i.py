from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common_test_i import (
    CommonTestIDataset,
    MultiClassConvNet,
    PolarClassifier,
    ResNet18Classifier,
    ResidualClassifier,
    SpectralClassifier,
    list_multiclass_files,
    maybe_limit_items,
    run_epoch,
    save_report,
    set_seed,
    stratified_split_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Common Test I multiclass baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data/common-test-i"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--model-type",
        choices=["conv", "residual", "polar", "spectral", "resnet18"],
        default="conv",
    )
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--combine-train-and-val", action="store_true")
    parser.add_argument("--input-pool", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--center-crop", type=int, default=0)
    parser.add_argument("--resize-to", type=int, default=0)
    parser.add_argument("--normalize-mode", choices=["none", "per_image_standardize", "imagenet"], default="none")
    parser.add_argument("--repeat-channels", type=int, default=1)
    parser.add_argument("--view-mode", choices=["image", "polar"], default="image")
    parser.add_argument("--polar-radius", type=int, default=72)
    parser.add_argument("--polar-height", type=int, default=80)
    parser.add_argument("--polar-width", type=int, default=96)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
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

    full_train_items = list_multiclass_files(args.data_root, "train")
    if args.combine_train_and_val:
        full_train_items = full_train_items + list_multiclass_files(args.data_root, "val")
    if args.validation_fraction > 0.0:
        split_train_items, split_val_items = stratified_split_items(
            full_train_items,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        train_items = maybe_limit_items(split_train_items, args.train_fraction, args.seed)
        val_items = maybe_limit_items(split_val_items, args.val_fraction, args.seed)
    else:
        train_items = maybe_limit_items(full_train_items, args.train_fraction, args.seed)
        val_items = maybe_limit_items(list_multiclass_files(args.data_root, "val"), args.val_fraction, args.seed)

    train_loader = DataLoader(
        CommonTestIDataset(
            train_items,
            augment=not args.disable_augment,
            center_crop=args.center_crop if args.center_crop > 0 else None,
            resize_to=args.resize_to if args.resize_to > 0 else None,
            normalize_mode=args.normalize_mode,
            view_mode=args.view_mode,
            polar_radius=args.polar_radius,
            polar_height=args.polar_height,
            polar_width=args.polar_width,
            cache_dir=args.cache_dir,
            repeat_channels=args.repeat_channels,
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
            view_mode=args.view_mode,
            polar_radius=args.polar_radius,
            polar_height=args.polar_height,
            polar_width=args.polar_width,
            cache_dir=args.cache_dir,
            repeat_channels=args.repeat_channels,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model_type == "spectral":
        model = SpectralClassifier(width=max(args.width, 24)).to(device)
    elif args.model_type == "resnet18":
        model = ResNet18Classifier(
            input_channels=args.repeat_channels,
            pretrained=args.pretrained,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "polar":
        model = PolarClassifier(width=max(args.width, 24)).to(device)
    elif args.model_type == "residual":
        model = ResidualClassifier(width=args.width).to(device)
    else:
        model = MultiClassConvNet(input_pool=args.input_pool, width=args.width).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        if args.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=args.warmup_epochs,
            )
            cosine_epochs = max(args.epochs - args.warmup_epochs, 1)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=1e-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs, 1),
                eta_min=1e-6,
            )

    best_val_auc = float("-inf")
    best_val_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
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
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} train_auc={train_metrics.macro_roc_auc:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} val_auc={val_metrics.macro_roc_auc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        improvement = val_metrics.macro_roc_auc - best_val_auc
        if improvement > 0.0:
            best_val_auc = val_metrics.macro_roc_auc
            best_val_metrics = val_metrics
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)
        if improvement > args.early_stopping_min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch} best_val_auc={best_val_auc:.4f}")
            break

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
            "validation_fraction": args.validation_fraction,
            "combine_train_and_val": args.combine_train_and_val,
            "input_pool": args.input_pool,
            "width": args.width,
            "center_crop": args.center_crop,
            "resize_to": args.resize_to,
            "normalize_mode": args.normalize_mode,
            "repeat_channels": args.repeat_channels,
            "view_mode": args.view_mode,
            "polar_radius": args.polar_radius,
            "polar_height": args.polar_height,
            "polar_width": args.polar_width,
            "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
            "label_smoothing": args.label_smoothing,
            "pretrained": args.pretrained,
            "dropout": args.dropout,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "disable_augment": args.disable_augment,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": str(device),
            "model_path": str(args.model_path),
            "best_epoch": best_epoch,
        },
        history=history,
        validation_metrics=best_val_metrics,
    )


if __name__ == "__main__":
    main()
