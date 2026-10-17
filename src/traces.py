"""Vector-strand renderer for per-content-frame traced strands (out/traces/).

The traces are produced by `tools/trace/fit.py` (inverse rendering against
ref/). This module only *renders* them, so the same code is used (with torch
autograd) by the fitter and (without) by the 1080p video render:

    .venv/bin/python -m src.traces                 # out/traces/ -> out/traces.mp4 (1920x1080)
    .venv/bin/python -m src.traces --score         # ... then tools/score.py
    .venv/bin/python -m src.traces --still 361     # one 1080p PNG

Frame model (all lengths in 960x540 "trace" px; render scale s = 2 at 1080p):

* strand = polyline vertices (x_j, y_j) with brightness b_j >= 0 and width
  u_j in [0, 2] (0 = sharp core, 1 = pre-blurred by soft_sigma[0], 2 = by
  soft_sigma[1]; mass is split linearly between the neighbouring layers).
  The curve between vertices is a monotone cubic
  Hermite (PCHIP) in y(x), so it is smooth at any resolution and never
  overshoots, and it is sampled by arc length (<= 0.5 render px per sample),
  so near-vertical spikes get continuous strokes with mass b * ds.
* each sample is splatted bilinearly into three core buffers by width;
  exposure E_c = sum_k A_kc * G(sigma_k) * sum_l G(soft_l) * core_l
  (the 3-Gaussian line kernel of docs/analysis.md, peak-normalised so a
  horizontal line of b = 1 gets E_peak = sum_k A_kc), computed with FFTs.
* plus a low-res haze layer (bicubic-upsampled grid, own RGB tint) for
  smudges/blobs/ghost sheets, added to E.
* output = 255 * (1 - exp(-E) - offset), clipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "out", "traces")
TW, TH = 960, 540                 # trace coordinate frame (= scorer resolution)
N_FRAMES = 540
SAMPLE_BUCKET = 1 << 15


def content_index(i: int) -> int:
    """Output frame i shows content frame c (source runs at 29.97 fps)."""
    return max(1, 2 * ((i + 1) // 2) - 1)


def default_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ look
DEFAULT_LOOK = {
    # 3-Gaussian line kernel (analysis 1.4), RGB per component
    "sigma": [1.0, 3.3, 7.6],
    "A": [[0.93, 2.46, 0.53], [0.43, 0.92, 0.21], [0.035, 0.143, 0.0]],
    "soft_sigma": [1.5, 4.0],     # extra pre-blur of the width-1 / width-2 core layers
    "offset": 2.0 / 255.0,
    "haze_rgb": [0.3, 1.0, 0.12],
    "haze_step": 8,               # haze grid cell (trace px)
}


def look_tensors(look: dict, device) -> dict:
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    return {"sigma": t(look["sigma"]), "A": t(look["A"]), "soft_sigma": t(look["soft_sigma"]),
            "offset": t(look["offset"]), "haze_rgb": t(look["haze_rgb"])}


# ------------------------------------------------------------------ strands
def pchip_slopes(x, y, first, last):
    """PCHIP tangents dy/dx at each vertex of concatenated polylines.

    first/last: bool tensors marking each strand's first/last vertex."""
    n = x.shape[0]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    delta = dy / dx.clamp(min=1e-3)
    valid = ~last[:-1]                       # segment j joins j and j+1 of one strand
    delta = torch.where(valid, delta, torch.zeros_like(delta))
    dl = torch.cat([delta[:1] * 0, delta])   # slope of segment left of vertex
    dr = torch.cat([delta, delta[:1] * 0])   # slope right of vertex
    same = (dl * dr) > 0
    hm = 2 * dl * dr / torch.where(same, dl + dr, torch.ones_like(dl))
    d = torch.where(same, hm, torch.zeros_like(dl))
    d = torch.where(first, dr, d)
    d = torch.where(last, dl, d)
    d = torch.where(first & last, torch.zeros_like(d), d)
    return d


def sample_strands(X, Y, B, U, first, last, scale: float, step: float = 0.5):
    """Arc-length samples of all strands. Returns (px, py, [m_0, m_1, m_2]) in
    trace px / trace-length mass units, one mass per width layer."""
    d = pchip_slopes(X, Y, first, last)
    seg = torch.nonzero(~last[:-1]).squeeze(1)
    x0, x1 = X[seg], X[seg + 1]
    y0, y1 = Y[seg], Y[seg + 1]
    d0, d1 = d[seg], d[seg + 1]
    h = x1 - x0
    with torch.no_grad():
        chord = torch.sqrt(h ** 2 + (y1 - y0) ** 2) * scale
        # the Hermite curve can be longer than the chord; 1.3x margin
        n = torch.clamp(torch.ceil(chord * 1.3 / step), 1, 4096).long()
        idx = torch.repeat_interleave(torch.arange(len(seg), device=X.device), n)
        start = torch.cumsum(n, 0) - n
        tot = int(n.sum().item())
        k = torch.arange(tot, device=X.device) - start[idx]
        t = (k.float() + 0.5) / n[idx].float()
        # pad to a bucketed length with zero-mass samples: a new tensor shape
        # every iteration makes MPS slow down steadily (graph cache growth)
        padn = -tot % SAMPLE_BUCKET
        valid = torch.ones(tot + padn, device=X.device)
        if padn:
            idx = torch.cat([idx, torch.zeros(padn, dtype=idx.dtype, device=X.device)])
            t = torch.cat([t, torch.full((padn,), 0.5, device=X.device)])
            valid[tot:] = 0
    nn_ = n[idx].float()
    h_ = h[idx]
    t2, t3 = t * t, t * t * t
    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
    g00, g10, g01, g11 = 6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t
    ya, yb, da, db = y0[idx], y1[idx], d0[idx], d1[idx]
    py = h00 * ya + h10 * h_ * da + h01 * yb + h11 * h_ * db
    dydt = g00 * ya + g10 * h_ * da + g01 * yb + g11 * h_ * db
    px = x0[idx] + t * h_
    ds = torch.sqrt(h_ ** 2 + dydt ** 2 + 1e-12) / nn_
    bs = B[seg][idx] * (1 - t) + B[seg + 1][idx] * t
    us = U[seg][idx] * (1 - t) + U[seg + 1][idx] * t
    m = bs * ds * valid
    w0 = (1 - us).clamp(min=0)
    w2 = (us - 1).clamp(min=0)
    return px, py, [m * w0, m * (1 - w0 - w2), m * w2]


def splat(px, py, m, Hs, Ws):
    """Bilinear splat of masses m at render-pixel coords (px, py) into [Hs, Ws]."""
    x0 = torch.floor(px)
    y0 = torch.floor(py)
    fx = px - x0
    fy = py - y0
    x0 = x0.long()
    y0 = y0.long()
    idx, val = [], []
    for ox, oy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + ox
        yi = y0 + oy
        ok = (xi >= 0) & (xi < Ws) & (yi >= 0) & (yi < Hs)
        idx.append(torch.where(ok, yi * Ws + xi, torch.zeros_like(xi)))
        val.append(m * w * ok)
    out = torch.zeros(Hs * Ws, dtype=m.dtype, device=m.device)
    out.index_add_(0, torch.cat(idx), torch.cat(val))
    return out.view(Hs, Ws)


class Canvas:
    """Padded render canvas at scale s with cached FFT frequency grids."""

    def __init__(self, scale: int = 2, pad: int = None, device=None):
        self.s = scale
        self.device = device or default_device()
        self.H, self.W = TH * scale, TW * scale
        self.pad = pad if pad is not None else int(math.ceil(3.2 * 7.6 * scale))
        Hp, Wp = self.H + 2 * self.pad, self.W + 2 * self.pad
        # FFT-friendly sizes
        self.Hp = _good_size(Hp)
        self.Wp = _good_size(Wp)
        fy = torch.fft.fftfreq(self.Hp, device=self.device)
        fx = torch.fft.rfftfreq(self.Wp, device=self.device)
        self.f2 = (fy[:, None] ** 2 + fx[None, :] ** 2).float()

    def to_px(self, x, y):
        s = self.s
        return x * s + (s - 1) / 2 + self.pad, y * s + (s - 1) / 2 + self.pad

    def gauss_hat(self, sigma):
        """FT of a peak-normalised (line-integral) Gaussian of sigma trace px."""
        S = sigma * self.s
        return math.sqrt(2 * math.pi) * self.s * S * torch.exp(-2 * math.pi ** 2 * S ** 2 * self.f2)

    def blur_hat(self, sigma):
        S = sigma * self.s
        return torch.exp(-2 * math.pi ** 2 * S ** 2 * self.f2)

    def crop(self, img):
        p = self.pad
        return img[..., p:p + self.H, p:p + self.W]


def _good_size(n):
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def exposure(cv: Canvas, strands: dict, look: dict, haze=None):
    """Per-channel exposure [3, H, W] (RGB) for concatenated strands.

    strands: dict of tensors X, Y, B, U, first, last (see sample_strands).
    haze: optional [h, w] non-negative grid (trace px / look haze_step cells)."""
    Hp, Wp = cv.Hp, cv.Wp
    E = torch.zeros(3, cv.H, cv.W, device=cv.device)
    if strands is not None and strands["X"].numel() > 1:
        px, py, ms = sample_strands(strands["X"], strands["Y"], strands["B"], strands["U"],
                                    strands["first"], strands["last"], cv.s)
        rx, ry = cv.to_px(px, py)
        Fl = torch.fft.rfft2(splat(rx, ry, ms[0], Hp, Wp))
        for l in (1, 2):
            Fl = Fl + torch.fft.rfft2(splat(rx, ry, ms[l], Hp, Wp)) * cv.blur_hat(look["soft_sigma"][l - 1])
        gh = [cv.gauss_hat(look["sigma"][k]) for k in range(look["sigma"].shape[0])]
        chans = []
        for c in range(3):
            K = sum(look["A"][k, c] * gh[k] for k in range(len(gh)))
            chans.append(torch.fft.irfft2(Fl * K, s=(Hp, Wp)))
        E = cv.crop(torch.stack(chans))
    if haze is not None:
        hz = F.interpolate(haze[None, None], size=(cv.H, cv.W), mode="bicubic",
                           align_corners=False)[0, 0].clamp(min=0)
        E = E + look["haze_rgb"].view(3, 1, 1) * hz
    return E


def tonemap(E, offset):
    """RGB 0..255 float (unquantised)."""
    return (255.0 * (1.0 - torch.exp(-E.clamp(min=0)) - offset)).clamp(0, 255)


def downscale(img, s):
    """Area downscale [C, H, W] by integer s (what the scorer's ffmpeg does)."""
    if s == 1:
        return img
    return F.avg_pool2d(img[None], s)[0]


# ------------------------------------------------------------------ IO
def frame_path(c: int, d: str = TRACE_DIR) -> str:
    return os.path.join(d, f"c{c:04d}.npz")


def save_trace(path: str, strands_np: list, haze: np.ndarray, info: dict = None):
    """strands_np: list of dicts with x, y, b, u arrays (trace px)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lens = np.array([len(s["x"]) for s in strands_np], np.int32)
    cat = lambda k: (np.concatenate([s[k] for s in strands_np]).astype(np.float32)
                     if strands_np else np.zeros(0, np.float32))
    np.savez_compressed(path, lens=lens, x=cat("x"), y=cat("y"), b=cat("b"), u=cat("u"),
                        haze=haze.astype(np.float32), info=json.dumps(info or {}))


def load_trace(path: str):
    d = np.load(path)
    lens = d["lens"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    strands = [{k: d[k][offs[i]:offs[i + 1]] for k in ("x", "y", "b", "u")} for i in range(len(lens))]
    return strands, d["haze"], json.loads(str(d["info"]))


def pack(strands_np: list, device) -> dict:
    """List of numpy strand dicts -> concatenated tensors for exposure()."""
    if not strands_np:
        return None
    t = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=device)
    first = np.concatenate([np.arange(len(s["x"])) == 0 for s in strands_np])
    last = np.concatenate([np.arange(len(s["x"])) == len(s["x"]) - 1 for s in strands_np])
    return {"X": t([s["x"] for s in strands_np]), "Y": t([s["y"] for s in strands_np]),
            "B": t([s["b"] for s in strands_np]), "U": t([s["u"] for s in strands_np]),
            "first": torch.tensor(first, device=device), "last": torch.tensor(last, device=device)}


def load_look(d: str = TRACE_DIR) -> dict:
    p = os.path.join(d, "look.json")
    if os.path.exists(p):
        with open(p) as f:
            return {**DEFAULT_LOOK, **json.load(f)}
    return dict(DEFAULT_LOOK)


def render_trace(c: int, cv: Canvas, look_t: dict, d: str = TRACE_DIR) -> np.ndarray:
    """BGR uint8 frame at the canvas resolution for content frame c."""
    strands, haze, info = load_trace(frame_path(c, d))
    if "look" in info:                       # per-frame look (fit with --fit-look)
        look_t = look_tensors(info["look"], cv.device)
    with torch.no_grad():
        E = exposure(cv, pack(strands, cv.device), look_t,
                     torch.tensor(haze, device=cv.device) if haze.size else None)
        img = tonemap(E, look_t["offset"])
    rgb = img.permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(np.clip(rgb[:, :, ::-1] + 0.5, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=TRACE_DIR)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "traces.mp4"))
    ap.add_argument("--scale", type=int, default=2, help="2 = 1920x1080, 1 = 960x540, 4 = 3840x2160")
    ap.add_argument("--codec", default="libx264", choices=["libx264", "libsvtav1"])
    ap.add_argument("--preset", help="encoder preset (x264 name, or SVT-AV1 0-13; default slow / 4)")
    ap.add_argument("--crf", type=int, help="crf (default: x264 8, ~0.002 SSIM better than 16; SVT-AV1 18)")
    ap.add_argument("--still", type=int, help="render one output frame to PNG")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-compensate", action="store_true",
                    help="disable the encoder level-offset pre-compensation")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    import cv2
    from .encode import Mp4Writer, PngDirWriter
    dev = torch.device("cpu") if args.cpu else default_device()
    cv = Canvas(args.scale, device=dev)
    look_t = look_tensors(load_look(args.traces), dev)
    if args.still is not None:
        out = os.path.join(ROOT, "out", f"trace_still_{args.still:04d}.png")
        cv2.imwrite(out, render_trace(content_index(args.still), cv, look_t, args.traces))
        print(out)
        return
    t0 = time.time()
    if args.out.endswith(".mp4"):
        av1 = args.codec == "libsvtav1"
        crf = args.crf if args.crf is not None else (18 if av1 else 8)
        preset = args.preset or ("4" if av1 else "slow")
        w = Mp4Writer(args.out, cv.W, cv.H, 60000, 1001, crf=crf, preset=preset,
                      compensate=not args.no_compensate, codec=args.codec)
    else:
        w = PngDirWriter(args.out)
    cache = {}
    for i in range(N_FRAMES):
        c = content_index(i)
        if c not in cache:
            cache = {c: render_trace(c, cv, look_t, args.traces)}
        w.write(cache[c])
        if i % 60 == 0:
            print(f"\r  {i}/{N_FRAMES}  {time.time() - t0:.1f}s", end="", file=sys.stderr)
    w.close()
    print(f"\rrendered {N_FRAMES} frames at {cv.W}x{cv.H} in {time.time() - t0:.1f}s -> {args.out}",
          file=sys.stderr)
    if args.score:
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "score.py"), args.out], check=True)


if __name__ == "__main__":
    main()
