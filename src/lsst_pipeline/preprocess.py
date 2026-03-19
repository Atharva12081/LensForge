from __future__ import annotations

import numpy as np


def percentile_normalize(array: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected (C, H, W) input, got {array.shape}.")
    output = np.empty_like(array, dtype=np.float32)
    for channel_idx, channel in enumerate(array):
        lo = float(np.percentile(channel, lower))
        hi = float(np.percentile(channel, upper))
        if hi <= lo:
            output[channel_idx] = np.clip(channel, 0.0, 1.0).astype(np.float32, copy=False)
            continue
        output[channel_idx] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    return output
