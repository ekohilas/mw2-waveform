"""Differentiable torch re-implementation of tools/score.py's per-frame metrics.

ssim_map() matches skimage.metrics.structural_similarity with its defaults
(7x7 uniform window, scipy 'reflect' borders, sample covariance, K1 0.01,
K2 0.03, data_range 255); `python -m tools.trace.metrics` validates it (and
the other three metrics) against tools/score.py on reference / render pairs.
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2
LUMA = (0.299, 0.587, 0.114)     # RGB weights (cv2 BGR2GRAY)


def luma(rgb):
    """[3, H, W] RGB -> [H, W]."""
    return LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]


def _sym_pad(x, p):
    """scipy.ndimage 'reflect' (symmetric, edge repeated) padding of [N, C, H, W]."""
    x = torch.cat([x[..., :p, :].flip(-2), x, x[..., -p:, :].flip(-2)], -2)
    return torch.cat([x[..., :p].flip(-1), x, x[..., -p:].flip(-1)], -1)


def uniform7(x):
    """7x7 mean filter, [N, C, H, W], symmetric borders (separable)."""
    p = 3
    xp = _sym_pad(x, p)
    k = torch.full((1, 1, 1, 7), 1 / 7, device=x.device, dtype=x.dtype)
    C = x.shape[1]
    y = F.conv2d(xp.reshape(-1, 1, *xp.shape[-2:]), k)
    y = F.conv2d(y, k.transpose(-1, -2))
    return y.reshape(x.shape)


def ssim_map(a, b):
    """SSIM map of two [H, W] luma images (0..255), skimage defaults."""
    np_ = 49
    cov = np_ / (np_ - 1)
    st = torch.stack([a, b, a * a, b * b, a * b])[None]
    m = uniform7(st)[0]
    ux, uy, uxx, uyy, uxy = m
    vx = cov * (uxx - ux * ux)
    vy = cov * (uyy - uy * uy)
    vxy = cov * (uxy - ux * uy)
    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux * ux + uy * uy + C1
    B2 = vx + vy + C2
    return (A1 * A2) / (B1 * B2)


def gauss1d(sigma, ksize, device):
    r = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-r ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _reflect101_pad(x, p):
    return F.pad(x, (p, p, p, p), mode="reflect")


def gblur(x, sigma, ksize):
    """cv2.GaussianBlur-like (BORDER_REFLECT_101) of [C, H, W]."""
    k = gauss1d(sigma, ksize, x.device)
    p = ksize // 2
    C = x.shape[0]
    xp = _reflect101_pad(x[None], p)[0]
    y = F.conv2d(xp[:, None], k.view(1, 1, 1, -1))
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y[:, 0]


def lit_mask(ly_ref_u8: np.ndarray, ly_out_u8: np.ndarray) -> np.ndarray:
    m = ((ly_ref_u8 > 8) | (ly_out_u8 > 8)).astype(np.uint8)
    return cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0


class RefFrame:
    """Reference frame tensors needed by the losses (precomputed once)."""

    def __init__(self, bgr: np.ndarray, device):
        self.bgr = bgr
        self.ly_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = bgr[:, :, ::-1].astype(np.float32)
        self.rgb = torch.tensor(rgb.copy(), device=device).permute(2, 0, 1)
        self.ly = torch.tensor(self.ly_u8.astype(np.float32), device=device)
        br = cv2.GaussianBlur(bgr, (0, 0), 6).astype(np.float32)[:, :, ::-1].copy()
        self.blur_rgb = torch.tensor(br, device=device).permute(2, 0, 1)
        self.blur_ly = luma(self.blur_rgb)
        e = self.ly_u8.astype(np.float32).sum(0)
        self.energy = torch.tensor(cv2.GaussianBlur(e.reshape(1, -1), (0, 0), 8).ravel(), device=device)
        self.device = device


def metrics(out_rgb, ref: RefFrame, lit: torch.Tensor):
    """Differentiable approximations of the four score.py metrics.

    out_rgb: [3, 540, 960] float 0..255. lit: bool [540, 960]."""
    ly = luma(out_rgb)
    smap = ssim_map(ref.ly, ly)
    n = lit.sum().clamp(min=1)
    ssim = (smap * lit).sum() / n
    bo = gblur(out_rgb, 6.0, 37)
    smap_b = ssim_map(ref.blur_ly, luma(bo))
    ssim_b = (smap_b * lit).sum() / n
    color = 1 - (bo - ref.blur_rgb).abs().sum() / torch.maximum(bo, ref.blur_rgb).sum().clamp(min=1e-6)
    e = ly.sum(0)
    k = gauss1d(8.0, 65, e.device)
    eb = F.conv1d(F.pad(e.view(1, 1, -1), (32, 32), mode="reflect"), k.view(1, 1, -1))[0, 0]
    energy = 1 - (eb - ref.energy).abs().sum() / torch.maximum(eb, ref.energy).sum().clamp(min=1e-6)
    return {"ssim": ssim, "ssim_blur": ssim_b, "color": color, "energy": energy}


def quantize_bgr(out_rgb):
    return np.clip(out_rgb.detach().permute(1, 2, 0).cpu().numpy()[:, :, ::-1] + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ validation
def _validate():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import score  # tools/score.py
    from skimage.metrics import structural_similarity
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dev = torch.device("cpu")
    rng = np.random.default_rng(0)
    pairs = []
    for i in (21, 101, 345, 381):
        ref = cv2.imread(os.path.join(root, "ref", "frames", f"{i:04d}.png"))
        oth = cv2.imread(os.path.join(root, "ref", "frames", f"{i + 6:04d}.png"))
        noisy = np.clip(ref.astype(np.float32) * 0.8 + rng.normal(0, 6, ref.shape), 0, 255).astype(np.uint8)
        pairs += [(ref, oth), (ref, noisy)]
    for ref, out in pairs:
        m_ref = score.frame_metrics(ref, out)
        r = RefFrame(ref, dev)
        ly_o = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        lit = torch.tensor(lit_mask(r.ly_u8, ly_o))
        o = torch.tensor(out[:, :, ::-1].astype(np.float32).copy()).permute(2, 0, 1)
        m = {k: float(v) for k, v in metrics(o, r, lit).items()}
        # exact-input SSIM check (same uint8 luma as the scorer)
        _, smap = structural_similarity(r.ly_u8, ly_o, data_range=255, full=True)
        t = ssim_map(r.ly, torch.tensor(ly_o.astype(np.float32)))
        print("skimage-vs-torch map max|d| %.2e  |  " % float(np.abs(smap - t.numpy()).max()) +
              "  ".join(f"{k} {m_ref[k]:.4f}/{m[k]:.4f}" for k in m))


if __name__ == "__main__":
    _validate()
