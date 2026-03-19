from __future__ import annotations

import numpy as np


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected (C, H, W) input, got {array.shape}.")
    _, height, width = array.shape
    crop_h = min(size, height)
    crop_w = min(size, width)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return array[:, top : top + crop_h, left : left + crop_w]
