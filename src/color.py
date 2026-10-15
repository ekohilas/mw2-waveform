"""Linear light -> display colour: tonemap then a tunable colour ramp."""
from __future__ import annotations

import numpy as np

from .params.schema import Look


def tonemap(L: np.ndarray, look: Look) -> np.ndarray:
    L = L * look.exposure
    if look.tonemap == "clip":
        return np.clip(L, 0.0, 1.0)
    return 1.0 - np.exp(-np.maximum(L, 0.0))


def ramp_lut(look: Look, n: int = 1024) -> np.ndarray:
    """[n, 3] float32 BGR lookup for v in [0, 1]."""
    st = np.asarray(look.ramp, dtype=np.float64)
    v = np.linspace(0, 1, n)
    rgb = np.stack([np.interp(v, st[:, 0], st[:, c]) for c in (1, 2, 3)], -1)
    return rgb[:, ::-1].astype(np.float32)  # BGR


def to_bgr8(L: np.ndarray, look: Look, lut: np.ndarray) -> np.ndarray:
    v = tonemap(L, look)
    n = lut.shape[0]
    idx = np.clip((v * (n - 1) + 0.5).astype(np.int32), 0, n - 1)
    out = lut[idx]
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
