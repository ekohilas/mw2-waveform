#!/usr/bin/env python3
"""Trace the dominant waveform strands in every content frame.

For each content frame (odd output frames 1..539) the luma image is reduced to
an "off-baseline ridge" map (baseline band |dy| < 4 attenuated), then a
dynamic-programming path y(x) that maximises summed luma minus an L1 slope
penalty is found (exact, via a 1-D L1 distance transform per column). The
path's pixels (+-5 px) are erased and the next strand is traced, K strands per
frame. Also stores the per-column luma-weighted signed centroid of the
off-baseline light.

Output (gitignored): out/audio/trace.npz
  frames  (270,)        output frame index of each content frame
  d       (270, K, 960) strand displacement above the baseline (px, +up)
  s       (270, K, 960) per-column ridge luma along the path
  cen     (270, 960)    signed luma-weighted centroid of off-baseline light

Usage: .venv/bin/python tools/audio/trace.py [--k 3]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Y0 = 269.5
H, W = 540, 960


def l1_dt(f, pen):
    """g[y] = max_y' f[y'] - pen*|y-y'| and its argmax (vectorised)."""
    n = len(f)
    y = np.arange(n)
    a = f + pen * y                      # y' <= y
    cm = np.maximum.accumulate(a)
    ia = np.maximum.accumulate(np.where(a >= cm, y, 0))
    left = cm - pen * y
    b = (f - pen * y)[::-1]              # y' >= y, reversed
    cmb = np.maximum.accumulate(b)
    ib = (n - 1 - np.maximum.accumulate(np.where(b >= cmb, y, 0)))[::-1]
    right = cmb[::-1] + pen * y
    use_l = left >= right
    return np.where(use_l, left, right), np.where(use_l, ia, ib)


def dp_path(L, pen):
    """Max-sum path through columns of L (H x W), L1 slope penalty."""
    Hh, Ww = L.shape
    acc = L[:, 0].astype(np.float64).copy()
    back = np.zeros((Hh, Ww), np.int32)
    for x in range(1, Ww):
        g, arg = l1_dt(acc, pen)
        acc = g + L[:, x]
        back[:, x] = arg
    y = np.zeros(Ww, int)
    y[-1] = int(np.argmax(acc))
    for x in range(Ww - 1, 0, -1):
        y[x - 1] = back[y[x], x]
    return y


def trace_frame(i, k=3, pen=2.0, floor=40.0):
    img = cv2.imread(os.path.join(ROOT, "ref", "frames", f"{i:04d}.png"))
    l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.arange(H)[:, None] - Y0
    att = np.where(np.abs(rows) < 4, 0.2, 1.0).astype(np.float32)
    Lm = cv2.GaussianBlur(l, (0, 0), 0.8) * att - floor   # dark costs, lit pays
    ds, ss = [], []
    for _ in range(k):
        y = dp_path(Lm, pen)
        ds.append(Y0 - y)
        ss.append(l[y, np.arange(W)])
        for x in range(W):
            Lm[max(0, y[x] - 5):y[x] + 6, x] = -floor
    off = l * (np.abs(rows) >= 5)
    off = np.maximum(off - 20, 0)
    cen = ((-rows) * off).sum(0) / np.maximum(off.sum(0), 1)
    return i, np.array(ds), np.array(ss), cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    frames = np.arange(1, 540, 2)
    with ProcessPoolExecutor() as ex:
        res = sorted(ex.map(trace_frame, frames, [a.k] * len(frames)))
    os.makedirs(os.path.join(ROOT, "out", "audio"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "out", "audio", "trace.npz"), frames=frames,
                        d=np.array([r[1] for r in res]), s=np.array([r[2] for r in res]),
                        cen=np.array([r[3] for r in res]))
    print("traced", len(frames), "frames")


if __name__ == "__main__":
    main()
