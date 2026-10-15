"""Rasterisation into float32 linear light buffers.

Strand cores
------------
Each strand is a polyline sampled at `supersample` points per output column.
Every segment deposits its light mass (brightness * thickness * arc length)
uniformly over the vertical span it covers in its column, widened by the
line thickness. The span is written exactly (fractional end pixels) with a
4-tap difference array per segment and resolved with one cumulative sum per
layer, so cost is O(segments + pixels) regardless of how tall a spike is.
Flat lines come out with constant brightness, steep lines keep the same
perceived brightness (mass ~ arc length).
"""
from __future__ import annotations

import cv2
import numpy as np


def raster_strands(y: np.ndarray, b: np.ndarray, thick: np.ndarray, xs: np.ndarray,
                   W: int, H: int) -> np.ndarray:
    """y, b: [S, N] in *output* pixel units; xs: [N] output px (ascending).

    Returns float32 [H, W] core buffer.
    """
    S, N = y.shape
    if S == 0:
        return np.zeros((H, W), np.float32)
    dx = np.diff(xs)                              # [N-1]
    xm = 0.5 * (xs[1:] + xs[:-1])
    col = np.floor(xm).astype(np.int64)
    okc = (col >= 0) & (col < W)
    y0, y1 = y[:, :-1], y[:, 1:]
    dy = y1 - y0
    arc = np.sqrt(dx[None, :] ** 2 + dy ** 2)
    th = thick[:, None]
    lo = np.minimum(y0, y1) - 0.5 * th
    hi = np.maximum(y0, y1) + 0.5 * th
    bm = 0.5 * (b[:, :-1] + b[:, 1:])
    mass = bm * th * arc
    v = mass / (hi - lo)
    lo = np.clip(lo, 0.0, H)
    hi = np.clip(hi, 0.0, H)
    keep = okc[None, :] & (hi > lo) & (v > 0)
    lo, hi, v = lo[keep], hi[keep], v[keep]
    c = np.broadcast_to(col[None, :], (S, N - 1))[keep]
    ka = np.floor(lo).astype(np.int64)
    fa = lo - ka
    kb = np.floor(hi).astype(np.int64)
    fb = hi - kb
    R = H + 2
    idx = np.concatenate([ka * W + c, (ka + 1) * W + c, kb * W + c, (kb + 1) * W + c])
    wts = np.concatenate([v * (1 - fa), v * fa, -v * (1 - fb), -v * fb])
    acc = np.bincount(idx, weights=wts, minlength=R * W)[:R * W].reshape(R, W)
    return np.cumsum(acc, axis=0)[:H].astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas are done on a downsampled copy."""
    if sigma <= 0.05:
        return img
    H, W = img.shape[:2]
    f = int(max(1, min(8, sigma // 3)))
    if f == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    small = cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f, borderType=cv2.BORDER_CONSTANT)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def add_gaussian_blob(buf: np.ndarray, x: float, y: float, sx: float, sy: float,
                      a: float, angle_deg: float) -> None:
    """Add an anisotropic Gaussian (output px) into buf in-place."""
    if a == 0 or sx <= 0 or sy <= 0:
        return
    H, W = buf.shape
    r = 3.5 * max(sx, sy)
    x0, x1 = int(max(0, x - r)), int(min(W, x + r + 1))
    y0, y1 = int(max(0, y - r)), int(min(H, y + r + 1))
    if x1 <= x0 or y1 <= y0:
        return
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5 - x, np.arange(y0, y1) + 0.5 - y)
    th = np.deg2rad(angle_deg)
    u = X * np.cos(th) + Y * np.sin(th)
    v = -X * np.sin(th) + Y * np.cos(th)
    buf[y0:y1, x0:x1] += (a * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))).astype(np.float32)
