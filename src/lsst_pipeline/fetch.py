from __future__ import annotations

from pathlib import Path

import numpy as np


def fetch_array(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array at {path}, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)
