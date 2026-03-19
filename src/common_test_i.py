from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


COMMON_TEST_I_LABELS = {"no": 0, "sphere": 1, "vort": 2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CommonTestIDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        items: list[tuple[Path, int]],
        augment: bool = False,
        center_crop: int | None = None,
        resize_to: int | None = None,
        normalize_mode: str = "none",
    ) -> None:
        self.items = items
        self.augment = augment
        self.center_crop = center_crop
        self.resize_to = resize_to
        self.normalize_mode = normalize_mode

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.items[index]
        image = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(image)
        if self.center_crop is not None:
            tensor = apply_center_crop(tensor, self.center_crop)
        if self.resize_to is not None:
            tensor = apply_resize(tensor, self.resize_to)
        tensor = apply_normalization(tensor, self.normalize_mode)
        if self.augment:
            tensor = apply_augmentation(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)


def apply_augmentation(image: Tensor) -> Tensor:
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(1,))
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(2,))
    if torch.rand(1).item() < 0.5:
        k = int(torch.randint(0, 4, (1,)).item())
        image = torch.rot90(image, k=k, dims=(1, 2))
    if torch.rand(1).item() < 0.3:
        image = torch.clamp(image + torch.randn_like(image) * 0.01, 0.0, 1.0)
    return image


def apply_center_crop(image: Tensor, crop_size: int) -> Tensor:
    _, height, width = image.shape
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return image[:, top : top + crop_size, left : left + crop_size]


def apply_resize(image: Tensor, size: int) -> Tensor:
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def apply_normalization(image: Tensor, mode: str) -> Tensor:
    if mode == "per_image_standardize":
        mean = image.mean()
        std = image.std().clamp_min(1e-6)
        return (image - mean) / std
    return image


def list_multiclass_files(data_root: Path, split: str) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for class_name, label in COMMON_TEST_I_LABELS.items():
        items.extend((path, label) for path in sorted((data_root / split / class_name).glob("*.npy")))
    return items


def maybe_limit_items(
    items: list[tuple[Path, int]],
    fraction: float,
    seed: int,
) -> list[tuple[Path, int]]:
    if fraction >= 1.0:
        return items
    rng = np.random.default_rng(seed)
    grouped: dict[int, list[tuple[Path, int]]] = {}
    for item in items:
        grouped.setdefault(item[1], []).append(item)
    kept: list[tuple[Path, int]] = []
    for label, label_items in grouped.items():
        take = max(1, int(len(label_items) * fraction))
        indices = rng.permutation(len(label_items))[:take]
        kept.extend(label_items[int(i)] for i in indices)
    return kept


class MultiClassConvNet(nn.Module):
    def __init__(self, num_classes: int = 3, input_pool: int = 1, width: int = 16) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if input_pool > 1:
            layers.append(nn.AvgPool2d(kernel_size=input_pool))
        self.features = nn.Sequential(
            *layers,
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 4, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 8, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


class ResidualClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, width: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(width),
            nn.MaxPool2d(2),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(width * 2),
            nn.MaxPool2d(2),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(width * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.proj2(x)
        x = self.stage2(x)
        x = self.proj3(x)
        x = self.stage3(x)
        return self.head(x)


@dataclass
class MultiClassMetrics:
    loss: float
    accuracy: float
    macro_roc_auc: float


def compute_macro_roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float32)[targets]
    return float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[MultiClassMetrics, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    probability_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probability_rows.append(probs)
        target_rows.append(labels.detach().cpu().numpy())
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    targets = np.concatenate(target_rows).astype(np.int32)
    probabilities = np.concatenate(probability_rows).astype(np.float32)
    predictions = probabilities.argmax(axis=1)
    metrics = MultiClassMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=float((predictions == targets).mean()),
        macro_roc_auc=compute_macro_roc_auc(targets, probabilities),
    )
    return metrics, targets, probabilities


def save_report(
    report_path: Path,
    config: dict[str, object],
    history: list[dict[str, object]],
    validation_metrics: MultiClassMetrics,
) -> None:
    payload = {
        "config": config,
        "history": history,
        "best_validation": asdict(validation_metrics),
    }
    report_path.write_text(json.dumps(payload, indent=2))
