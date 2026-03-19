from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LensDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, items: list[tuple[Path, int]], augment: bool = False) -> None:
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.items[index]
        image = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(image)
        if self.augment:
            tensor = apply_augmentation(tensor)
        target = torch.tensor(label, dtype=torch.float32)
        return tensor, target


def apply_augmentation(image: Tensor) -> Tensor:
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(1,))
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(2,))
    if torch.rand(1).item() < 0.5:
        k = int(torch.randint(0, 4, (1,)).item())
        image = torch.rot90(image, k=k, dims=(1, 2))
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(image) * 0.01
        image = torch.clamp(image + noise, 0.0, 1.0)
    return image


def list_labeled_files(data_root: Path, positive_dir: str, negative_dir: str) -> list[tuple[Path, int]]:
    positives = [(path, 1) for path in sorted((data_root / positive_dir).glob("*.npy"))]
    negatives = [(path, 0) for path in sorted((data_root / negative_dir).glob("*.npy"))]
    return positives + negatives


def split_train_validation(
    items: list[tuple[Path, int]],
    validation_fraction: float,
    seed: int,
    train_fraction: float,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    labels = [label for _, label in items]
    indices = np.arange(len(items))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        stratify=labels,
        random_state=seed,
    )
    if 0 < train_fraction < 1:
        train_labels = [labels[i] for i in train_idx]
        train_idx, _ = train_test_split(
            train_idx,
            train_size=train_fraction,
            stratify=train_labels,
            random_state=seed,
        )
    train_items = [items[int(i)] for i in train_idx]
    val_items = [items[int(i)] for i in val_idx]
    return train_items, val_items


def maybe_downsample_items(
    items: list[tuple[Path, int]],
    fraction: float,
    seed: int,
) -> list[tuple[Path, int]]:
    if fraction >= 1:
        return items

    labels = [label for _, label in items]
    indices = np.arange(len(items))
    keep_idx, _ = train_test_split(
        indices,
        train_size=fraction,
        stratify=labels,
        random_state=seed,
    )
    return [items[int(i)] for i in keep_idx]


def make_weighted_sampler(items: list[tuple[Path, int]]) -> WeightedRandomSampler:
    labels = np.array([label for _, label in items])
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(items),
        replacement=True,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class LensClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.10),
            ConvBlock(64, 128, dropout=0.15),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x)).squeeze(1)


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = focal_factor * bce
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_factor * loss
        return loss.mean()


@dataclass
class EpochMetrics:
    loss: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
    threshold: float = 0.5,
) -> EpochMetrics:
    predictions = (probabilities >= threshold).astype(np.int32)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    accuracy = float((predictions == targets).mean())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    unique_targets = np.unique(targets)
    roc_auc = float(roc_auc_score(targets, probabilities)) if len(unique_targets) > 1 else math.nan
    pr_auc = (
        float(average_precision_score(targets, probabilities))
        if len(unique_targets) > 1
        else math.nan
    )
    return EpochMetrics(
        loss=loss,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        threshold=threshold,
    )


def find_best_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = (probabilities >= threshold).astype(np.int32)
        tp = int(((predictions == 1) & (targets == 1)).sum())
        fp = int(((predictions == 1) & (targets == 0)).sum())
        fn = int(((predictions == 0) & (targets == 1)).sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2 * precision * recall) / max(precision + recall, 1e-8))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(targets).astype(np.int32)
    y_prob = np.concatenate(probabilities).astype(np.float32)
    mean_loss = total_loss / max(total_examples, 1)
    return compute_metrics(y_true, y_prob, mean_loss, threshold=threshold)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())

    y_true = np.concatenate(targets).astype(np.int32)
    y_prob = np.concatenate(probabilities).astype(np.float32)
    mean_loss = total_loss / max(total_examples, 1)
    return compute_metrics(y_true, y_prob, mean_loss, threshold=threshold), y_true, y_prob


def create_data_loaders(
    data_root: Path,
    batch_size: int,
    seed: int,
    validation_fraction: float,
    train_fraction: float,
    test_fraction: float,
    num_workers: int,
    balance_strategy: str,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    float,
    list[tuple[Path, int]],
    list[tuple[Path, int]],
]:
    train_items_all = list_labeled_files(data_root, "train_lenses", "train_nonlenses")
    train_items, val_items = split_train_validation(
        train_items_all,
        validation_fraction=validation_fraction,
        seed=seed,
        train_fraction=train_fraction,
    )
    test_items = maybe_downsample_items(
        list_labeled_files(data_root, "test_lenses", "test_nonlenses"),
        fraction=test_fraction,
        seed=seed,
    )

    use_sampler = balance_strategy in {"sampler", "both"}
    train_loader = DataLoader(
        LensDataset(train_items, augment=True),
        batch_size=batch_size,
        sampler=make_weighted_sampler(train_items) if use_sampler else None,
        shuffle=not use_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        LensDataset(val_items, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        LensDataset(test_items, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_labels = np.array([label for _, label in train_items], dtype=np.int32)
    class_counts = np.bincount(train_labels, minlength=2)
    pos_weight = float(class_counts[0] / max(class_counts[1], 1))
    return train_loader, val_loader, test_loader, pos_weight, val_items, test_items


def metrics_to_dict(metrics: EpochMetrics) -> dict[str, float | int]:
    return asdict(metrics)


def build_prediction_rows(
    items: list[tuple[Path, int]],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (path, label), probability in zip(items, probabilities, strict=True):
        predicted_label = int(probability >= threshold)
        if label == 1 and predicted_label == 1:
            outcome = "tp"
        elif label == 0 and predicted_label == 1:
            outcome = "fp"
        elif label == 0 and predicted_label == 0:
            outcome = "tn"
        else:
            outcome = "fn"
        rows.append(
            {
                "path": str(path),
                "label": int(label),
                "probability": float(probability),
                "predicted_label": predicted_label,
                "threshold": threshold,
                "outcome": outcome,
            }
        )
    return rows


def build_curve_payload(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(targets, probabilities)
    return {
        "roc_curve": {
            "fpr": fpr.astype(float).tolist(),
            "tpr": tpr.astype(float).tolist(),
            "thresholds": roc_thresholds.astype(float).tolist(),
        },
        "pr_curve": {
            "precision": precision.astype(float).tolist(),
            "recall": recall.astype(float).tolist(),
            "thresholds": pr_thresholds.astype(float).tolist(),
        },
    }


def save_training_report(
    report_path: Path,
    history: list[dict[str, object]],
    validation_metrics: EpochMetrics,
    test_metrics: EpochMetrics,
    config: dict[str, object],
    validation_curves: dict[str, list[float]] | None = None,
    test_curves: dict[str, list[float]] | None = None,
    validation_predictions: list[dict[str, object]] | None = None,
    test_predictions: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "config": config,
        "history": history,
        "best_validation": metrics_to_dict(validation_metrics),
        "test": metrics_to_dict(test_metrics),
    }
    if validation_curves is not None:
        payload["best_validation_curves"] = validation_curves
    if test_curves is not None:
        payload["test_curves"] = test_curves
    if validation_predictions is not None:
        payload["best_validation_predictions"] = validation_predictions
    if test_predictions is not None:
        payload["test_predictions"] = test_predictions
    report_path.write_text(json.dumps(payload, indent=2))
