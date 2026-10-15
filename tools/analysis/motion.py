#!/usr/bin/env python3
"""Dynamics: frame holds, global scroll test, regional drift, temporal
decorrelation, travelling 'needle' spikes and dim-layer persistence.

Usage: .venv/bin/python tools/analysis/motion.py
"""
import os
import sys

import cv2
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from common import H, N, W, load_all_luma, src_time  # noqa: E402

Y0 = 269.5
U = np.arange(1, N, 2)         # content frames (see holds)


def holds(L):
    d = np.array([np.abs(L[i] - L[i - 1]).mean() for i in range(1, N)])
    held = np.flatnonzero(d < 0.1) + 1
    print(f"frames identical to their predecessor: {len(held)} -> {held[:6].tolist()} ... {held[-3:].tolist()}")
    print(f"content changes between frames 2k and 2k+1; mean |dL| at changes p1/50/99/max ="
          f" {np.percentile(d[d >= 0.1], [1, 50, 99, 100]).round(2).tolist()} (no cut/flash outliers)")
    ml = L.mean((1, 2))
    print(f"frame mean luma range {ml.min():.2f}..{ml.max():.2f}; max change between content frames"
          f" {np.abs(np.diff(ml[U])).max():.2f}")


def hp(e):
    return e - cv2.GaussianBlur(e.reshape(1, -1).astype(np.float32), (0, 0), 40).ravel()


def global_scroll(L):
    win = cv2.createHanningWindow((W, 120), cv2.CV_32F)
    res = np.array([cv2.phaseCorrelate(np.ascontiguousarray(L[a, 210:330]),
                                       np.ascontiguousarray(L[b, 210:330]), win)[0]
                    for a, b in zip(U[:-1], U[1:])])
    print(f"\nglobal phase-correlation shift per content frame: dx p10/50/90 = "
          f"{np.percentile(res[:, 0], [10, 50, 90]).round(2).tolist()}, dy = "
          f"{np.percentile(res[:, 1], [10, 50, 90]).round(2).tolist()}")


def regional_drift(L):
    off = (np.abs(np.arange(H) - Y0) > 10)[:, None]
    out = {}
    for a, b in zip(U[:-1], U[1:]):
        A = np.where(off, L[a], 0)
        B = np.where(off, L[b], 0)
        for x0 in range(100, 760, 110):
            pa = A[:, x0:x0 + 220]
            if pa.sum() < 20000:
                continue
            best = max(((s, (pa * B[:, x0 + s:x0 + s + 220]).sum()
                         / np.sqrt((pa * pa).sum() * (B[:, x0 + s:x0 + s + 220] ** 2).sum() + 1e-9))
                        for s in range(-14, 15)), key=lambda z: z[1])
            out.setdefault(int(src_time(a)), []).append(best)
    print("\n220-px windows, best x-shift between consecutive content frames (px per content frame):")
    print("| second | shift p10/25/50/75/90 | best corr median |")
    print("|---|---|---|")
    for k in sorted(out):
        s = np.array(out[k])
        print(f"| {k} | {np.percentile(s[:, 0], [10, 25, 50, 75, 90]).tolist()} | {np.median(s[:, 1]):.2f} |")


def decorrelation(L):
    E = L.sum(1)
    print("\ntemporal correlation vs lag (content frames):")
    print("| lag (frames) | column-energy (hi-pass) | image, sigma 6 blur, off-baseline |")
    print("|---|---|---|")
    off = (np.abs(np.arange(H) - Y0) > 10)[:, None]
    B = [cv2.resize(cv2.GaussianBlur(np.where(off, L[i], 0), (0, 0), 6), (240, 135)).ravel() for i in U]
    for k in [1, 2, 3, 4, 8, 15, 30]:
        ce = np.median([np.corrcoef(hp(E[U[j]]), hp(E[U[j + k]]))[0, 1] for j in range(len(U) - k)])
        cb = np.median([np.corrcoef(B[j], B[j + k])[0, 1] for j in range(len(U) - k)])
        print(f"| {2 * k} | {ce:.2f} | {cb:.2f} |")


def needles(L):
    """Tall (>90 px), narrow (<10 px half-width) single spikes."""
    ys = np.abs(np.arange(H) - Y0)[:, None]
    hits = []
    for i in U:
        a = np.where(L[i] > 30, ys, 0).max(0)
        p, pr = find_peaks(a, height=90, width=(None, 10), prominence=60)
        hits += [(i, x, a[x]) for x in p]
    hits = np.array(hits, float)
    print(f"\nneedle spikes: {len(hits)} detections in {len(set(hits[:, 0]))} content frames")
    for name, sel in [("left-moving", (hits[:, 0] >= 339) & (hits[:, 0] <= 377) & (hits[:, 1] < 470)),
                      ("right-moving", (hits[:, 0] >= 339) & (hits[:, 0] <= 371) & (hits[:, 1] > 485))]:
        h = hits[sel]
        v, c = np.polyfit(h[:, 0], h[:, 1], 1)
        print(f"  {name} needle (frames {int(h[0, 0])}-{int(h[-1, 0])}): x = {c:.0f} + {v:.2f}*frame "
              f"(x={c + v * 339:.0f} at f339), amp {h[:, 2].max():.0f}->{h[-3:, 2].mean():.0f} px")


def dim_layer(L):
    off = (np.abs(np.arange(H) - Y0) > 12)[:, None]
    D, Es = [], []
    for i in U:
        l = L[i]
        m = off & (l > 3) & (cv2.dilate(l, np.ones((5, 5))) < 50)
        D.append(cv2.resize(np.where(m, l, 0), (240, 135), interpolation=cv2.INTER_AREA).ravel())
        Es.append(np.where(m, l, 0).sum() / (W * H))
    Es = np.array(Es)
    print("\ndim 'ghost' layer (off-baseline luma 3-50, away from bright strands):")
    print("  mean luma contribution per second:",
          {12 + s: round(float(Es[[int(src_time(i)) == 12 + s for i in U]].mean()), 2) for s in range(9)})
    print("  autocorr at lag 1/2/4/8/15 content frames:",
          [round(float(np.median([np.corrcoef(D[j], D[j + k])[0, 1] for j in range(len(U) - k)])), 2)
           for k in (1, 2, 4, 8, 15)])


def main():
    L = load_all_luma()
    holds(L)
    global_scroll(L)
    regional_drift(L)
    decorrelation(L)
    needles(L)
    dim_layer(L)


if __name__ == "__main__":
    main()
