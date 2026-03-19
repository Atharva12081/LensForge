from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from torch import nn

from src.lens_finding_baseline import (
    BinaryFocalLoss,
    build_curve_payload,
    build_prediction_rows,
    LensClassifier,
    create_data_loaders,
    find_best_threshold,
    metrics_to_dict,
    predict_loader,
    run_epoch,
    save_training_report,
    set_seed,
)


def default_data_root() -> Path:
    return Path("data/lens-finding-test")


def configure_matplotlib() -> None:
    tmp_root = Path("tmp").resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    mpl_dir = (tmp_root / "matplotlib").resolve()
    cache_dir = (tmp_root / "cache").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def save_curve_plots(
    report_path: Path,
    validation_curves: dict[str, list[float]],
    test_curves: dict[str, list[float]],
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(validation_curves["roc_curve"]["fpr"], validation_curves["roc_curve"]["tpr"], label="validation")
    axes[0].plot(test_curves["roc_curve"]["fpr"], test_curves["roc_curve"]["tpr"], label="test")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot(
        validation_curves["pr_curve"]["recall"],
        validation_curves["pr_curve"]["precision"],
        label="validation",
    )
    axes[1].plot(test_curves["pr_curve"]["recall"], test_curves["pr_curve"]["precision"], label="test")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(report_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "probability", "predicted_label", "threshold", "outcome"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline lens finder for DeepLense Test V.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-function", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument(
        "--balance-strategy",
        choices=["none", "sampler", "loss", "both"],
        default="both",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, test_loader, pos_weight, val_items, test_items = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
        balance_strategy=args.balance_strategy,
    )

    model = LensClassifier().to(device)
    use_pos_weight = args.balance_strategy in {"loss", "both"}
    pos_weight_tensor = torch.tensor(pos_weight, device=device) if use_pos_weight else None
    if args.loss_function == "focal":
        criterion = BinaryFocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            pos_weight=pos_weight_tensor,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_auc = float("-inf")
    best_val_metrics = None
    best_threshold = 0.5
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics_default, val_targets, val_probabilities = predict_loader(
            model, val_loader, criterion, device, threshold=0.5
        )
        threshold, _ = find_best_threshold(val_targets, val_probabilities)
        val_metrics_tuned, _, _ = predict_loader(
            model, val_loader, criterion, device, threshold=threshold
        )
        history.append(
            {
                "epoch": epoch,
                "train": metrics_to_dict(train_metrics),
                "validation": metrics_to_dict(val_metrics_default),
                "validation_tuned": metrics_to_dict(val_metrics_tuned),
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} train_auc={train_metrics.roc_auc:.4f} "
            f"val_loss={val_metrics_default.loss:.4f} val_auc={val_metrics_default.roc_auc:.4f} "
            f"val_pr_auc={val_metrics_default.pr_auc:.4f} "
            f"best_threshold={threshold:.2f}"
        )

        if val_metrics_default.roc_auc > best_val_auc:
            best_val_auc = val_metrics_default.roc_auc
            best_val_metrics = val_metrics_tuned
            best_threshold = threshold
            torch.save(model.state_dict(), args.model_path)

    if best_val_metrics is None:
        raise RuntimeError("Training did not produce validation metrics.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    best_validation_metrics, val_targets, val_probabilities = predict_loader(
        model, val_loader, criterion, device, threshold=best_threshold
    )
    test_metrics, test_targets, test_probabilities = predict_loader(
        model, test_loader, criterion, device, threshold=best_threshold
    )
    validation_curves = build_curve_payload(val_targets, val_probabilities)
    test_curves = build_curve_payload(test_targets, test_probabilities)
    validation_predictions = build_prediction_rows(val_items, val_probabilities, best_threshold)
    test_predictions = build_prediction_rows(test_items, test_probabilities, best_threshold)
    print(
        f"test_loss={test_metrics.loss:.4f} "
        f"test_auc={test_metrics.roc_auc:.4f} "
        f"test_pr_auc={test_metrics.pr_auc:.4f} "
        f"test_recall={test_metrics.recall:.4f}"
    )

    save_training_report(
        report_path=args.report_path,
        history=history,
        validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
        config={
            "data_root": str(args.data_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss_function": args.loss_function,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "validation_fraction": args.validation_fraction,
            "train_fraction": args.train_fraction,
            "test_fraction": args.test_fraction,
            "balance_strategy": args.balance_strategy,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": str(device),
            "pos_weight": pos_weight,
            "best_threshold": best_threshold,
            "model_path": str(args.model_path),
        },
        validation_curves=validation_curves,
        test_curves=test_curves,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
    )
    save_prediction_csv(args.report_path.with_name(f"{args.report_path.stem}_validation_predictions.csv"), validation_predictions)
    save_prediction_csv(args.report_path.with_name(f"{args.report_path.stem}_test_predictions.csv"), test_predictions)
    try:
        save_curve_plots(args.report_path, validation_curves, test_curves)
    except Exception as exc:
        print(f"plot_export_warning={exc}")


if __name__ == "__main__":
    main()
