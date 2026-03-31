from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
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
        view_mode: str = "image",
        polar_radius: int = 72,
        polar_height: int = 80,
        polar_width: int = 96,
        cache_dir: Path | None = None,
        repeat_channels: int = 1,
    ) -> None:
        self.items = items
        self.augment = augment
        self.center_crop = center_crop
        self.resize_to = resize_to
        self.normalize_mode = normalize_mode
        self.view_mode = view_mode
        self.polar_radius = polar_radius
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.cache_dir = cache_dir
        self.repeat_channels = repeat_channels
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.items[index]
        image = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(image)
        if self.center_crop is not None:
            tensor = apply_center_crop(tensor, self.center_crop)
        if self.view_mode == "polar":
            tensor = self._load_or_build_polar_view(path, tensor)
        if self.resize_to is not None:
            tensor = apply_resize(tensor, self.resize_to)
        if self.repeat_channels > 1 and tensor.shape[0] == 1:
            tensor = tensor.repeat(self.repeat_channels, 1, 1)
        tensor = apply_normalization(tensor, self.normalize_mode)
        if self.augment:
            if self.view_mode == "polar":
                tensor = apply_polar_augmentation(tensor)
            else:
                tensor = apply_augmentation(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

    def _load_or_build_polar_view(self, path: Path, tensor: Tensor) -> Tensor:
        if self.cache_dir is None:
            return apply_polar_view(
                tensor,
                radius=self.polar_radius,
                height=self.polar_height,
                width=self.polar_width,
            )
        cache_name = (
            f"{path.parent.name}_{path.stem}_r{self.polar_radius}_"
            f"h{self.polar_height}_w{self.polar_width}.npy"
        )
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            return torch.from_numpy(np.load(cache_path).astype(np.float32))
        polar = apply_polar_view(
            tensor,
            radius=self.polar_radius,
            height=self.polar_height,
            width=self.polar_width,
        )
        np.save(cache_path, polar.cpu().numpy().astype(np.float32))
        return polar


def apply_augmentation(image: Tensor) -> Tensor:
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(1,))
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(2,))
    if torch.rand(1).item() < 0.5:
        k = int(torch.randint(0, 4, (1,)).item())
        image = torch.rot90(image, k=k, dims=(1, 2))
    if torch.rand(1).item() < 0.3:
        max_shift_y = max(1, int(round(image.shape[1] * 0.05)))
        max_shift_x = max(1, int(round(image.shape[2] * 0.05)))
        shift_y = int(torch.randint(-max_shift_y, max_shift_y + 1, (1,)).item())
        shift_x = int(torch.randint(-max_shift_x, max_shift_x + 1, (1,)).item())
        image = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
    if torch.rand(1).item() < 0.3:
        image = torch.clamp(image + torch.randn_like(image) * 0.01, 0.0, 1.0)
    return image


def apply_polar_augmentation(image: Tensor) -> Tensor:
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=(2,))
    if torch.rand(1).item() < 0.5:
        shift = int(torch.randint(0, max(image.shape[2], 1), (1,)).item())
        image = torch.roll(image, shifts=shift, dims=2)
    if torch.rand(1).item() < 0.3:
        image = torch.clamp(image + torch.randn_like(image) * 0.01, -5.0, 5.0)
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


def apply_polar_view(image: Tensor, radius: int, height: int, width: int) -> Tensor:
    resize, warp_polar = load_skimage_transforms()
    base = image.squeeze(0).cpu().numpy().astype(np.float32)
    polar = warp_polar(
        base,
        center=((base.shape[0] - 1) / 2, (base.shape[1] - 1) / 2),
        radius=radius,
        output_shape=(height, width),
        scaling="linear",
        preserve_range=True,
    )
    polar = np.log1p(np.clip(polar, 0.0, None))
    polar = resize(
        polar,
        (height, width),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    if polar.shape != (height, width):
        polar = polar.T
    if polar.shape != (height, width):
        raise ValueError(f"Unexpected polar shape {polar.shape}, expected {(height, width)}.")
    return torch.from_numpy(polar[None])


@lru_cache(maxsize=1)
def load_skimage_transforms():
    try:
        from skimage.transform import resize, warp_polar
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Polar-view Common Test I models require scikit-image. "
            "Install `scikit-image` to use `--view-mode polar`."
        ) from exc
    return resize, warp_polar


@lru_cache(maxsize=1)
def load_torchvision_models():
    try:
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ResNet18 Common Test I models require torchvision. "
            "Install `torchvision` to use `--model-type resnet18`."
        ) from exc
    return models


def apply_normalization(image: Tensor, mode: str) -> Tensor:
    if mode == "per_image_standardize":
        mean = image.mean()
        std = image.std().clamp_min(1e-6)
        return (image - mean) / std
    if mode == "imagenet":
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if image.shape[0] == 1:
            mean = mean[:1]
            std = std[:1]
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


def stratified_split_items(
    items: list[tuple[Path, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    grouped: dict[int, list[tuple[Path, int]]] = {}
    for item in items:
        grouped.setdefault(item[1], []).append(item)
    train_items: list[tuple[Path, int]] = []
    validation_items: list[tuple[Path, int]] = []
    for label_items in grouped.values():
        indices = rng.permutation(len(label_items))
        validation_count = max(1, int(round(len(label_items) * validation_fraction)))
        validation_ids = set(int(idx) for idx in indices[:validation_count])
        for idx, item in enumerate(label_items):
            if idx in validation_ids:
                validation_items.append(item)
            else:
                train_items.append(item)
    return train_items, validation_items


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


class PolarClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, width: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
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
            nn.Conv2d(width * 4, width * 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 6),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 6, width * 4),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def complex_multiply_2d(input_ft: Tensor, weights: Tensor) -> Tensor:
    return torch.einsum("bixy,ioxy->boxy", input_ft, weights)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_height: int = 16, modes_width: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.top_weights = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat)
        )
        self.bottom_weights = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes_height, modes_width, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        width_ft = width // 2 + 1
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width_ft,
            dtype=torch.cfloat,
            device=x.device,
        )
        mh = min(self.modes_height, height)
        mw = min(self.modes_width, width_ft)
        out_ft[:, :, :mh, :mw] = complex_multiply_2d(x_ft[:, :, :mh, :mw], self.top_weights[:, :, :mh, :mw])
        out_ft[:, :, -mh:, :mw] = complex_multiply_2d(
            x_ft[:, :, -mh:, :mw],
            self.bottom_weights[:, :, :mh, :mw],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class SpectralClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, width: int = 24, modes_height: int = 16, modes_width: int = 16) -> None:
        super().__init__()
        width = max(width, 16)
        self.input_proj = nn.Conv2d(1, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes_height=modes_height, modes_width=modes_width) for _ in range(4)]
        )
        self.pointwise_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(4)])
        self.norm_layers = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(4)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        for spectral, pointwise, norm in zip(self.spectral_layers, self.pointwise_layers, self.norm_layers):
            x = F.gelu(norm(spectral(x) + pointwise(x)))
        return self.head(x)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 3, input_channels: int = 1, pretrained: bool = False, dropout: float = 0.2) -> None:
        super().__init__()
        models = load_torchvision_models()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        if input_channels != 3:
            original = self.backbone.conv1
            conv1 = nn.Conv2d(
                input_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=False,
            )
            with torch.no_grad():
                if pretrained:
                    base_weight = original.weight.detach()
                    if input_channels == 1:
                        conv1.weight.copy_(base_weight.mean(dim=1, keepdim=True))
                    else:
                        conv1.weight.copy_(base_weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1))
                else:
                    nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.conv1 = conv1
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


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
