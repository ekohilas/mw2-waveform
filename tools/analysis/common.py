"""Shared helpers for the reference analysis scripts (run with the repo venv)."""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_DIR = os.path.join(ROOT, "ref", "frames")
OUT_DIR = os.path.join(ROOT, "out", "analysis")
W, H, N = 960, 540, 540
FPS = 60000 / 1001


def frame_path(i):
    return os.path.join(REF_DIR, f"{i:04d}.png")


def load(i, rgb=True):
    """Frame i as float32 RGB (0..255)."""
    img = cv2.imread(frame_path(i))
    if rgb:
        img = img[:, :, ::-1]
    return img.astype(np.float32)


def luma(rgb):
    """Rec.601 luma of an RGB float image (matches cv2 BGR2GRAY used by score.py)."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def load_all_luma(step=1):
    """Stack of luma frames, shape (n, H, W) float32. Cached in out/analysis."""
    cache = os.path.join(OUT_DIR, f"luma_step{step}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    os.makedirs(OUT_DIR, exist_ok=True)
    arr = np.stack([luma(load(i)) for i in range(0, N, step)]).astype(np.float32)
    np.save(cache, arr)
    return arr


def src_time(i):
    return 12 + i * 1001 / 60000
