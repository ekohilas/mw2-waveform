#!/usr/bin/env python3
"""Detect soft, blurry 'smudge' blobs (low fine-detail, off the baseline) and
track them over time.

A pixel is smudge-like when its sigma=5 blurred luma > 10 and the local
high-frequency energy (|L - blur1.5(L)| averaged at sigma 5) is < 12% of it.
Components with area >= 120 px and |y - Y0| > 15 at their centroid are kept.

Usage: .venv/bin/python tools/analysis/smudges.py
Writes out/analysis/smudges.csv (per-frame blobs) and prints tracks.
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import H, N, OUT_DIR, load, load_all_luma, src_time  # noqa: E402

Y0 = 269.5


def main():
    L = load_all_luma()
    rows = []
    for i in range(1, N, 2):          # unique (content) frames only
        l = L[i]
        b = cv2.GaussianBlur(l, (0, 0), 5)
        hf = cv2.GaussianBlur(np.abs(l - cv2.GaussianBlur(l, (0, 0), 1.5)), (0, 0), 5)
        m = ((b > 10) & (hf / (b + 1) < 0.12)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        rgb = None
        for k in range(1, n):
            x, y, w, h, a = st[k]
            cx, cy = cen[k]
            if a < 120 or abs(cy - Y0) < 15:
                continue
            if rgb is None:
                rgb = load(i)
            sel = lab == k
            col = rgb[sel].mean(0)
            # blur estimate: mean gradient magnitude of the blurred-free luma at the blob
            gy, gx = np.gradient(l)
            g = np.hypot(gx, gy)[sel].mean()
            rows.append(dict(frame=i, t=round(src_time(i), 3), cx=round(cx, 1), cy=round(cy, 1),
                             x0=x, y0=y, w=w, h=h, area=a, peak=round(float(l[sel].max()), 1),
                             mean=round(float(l[sel].mean()), 1), R=round(col[0], 1),
                             G=round(col[1], 1), B=round(col[2], 1), grad=round(float(g), 2)))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "smudges.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} smudge blobs in {len(set(r['frame'] for r in rows))} of {N // 2} content frames")

    # link into tracks: nearest blob in next content frames within 60 px
    tracks = []
    for r in sorted(rows, key=lambda r: r["frame"]):
        best = None
        for tr in tracks:
            last = tr[-1]
            if 0 < r["frame"] - last["frame"] <= 6:
                d = np.hypot(r["cx"] - last["cx"], r["cy"] - last["cy"])
                if d < 60 and (best is None or d < best[0]):
                    best = (d, tr)
        if best:
            best[1].append(r)
        else:
            tracks.append([r])
    tracks = [t for t in tracks if len(t) >= 3]
    print("\n| frames | t (s) | x path | y centre | size w x h (max) | peak / mean luma | mean RGB | vx (px/frame) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in tracks:
        f = np.array([r["frame"] for r in t])
        x = np.array([r["cx"] for r in t])
        vx = np.polyfit(f, x, 1)[0] if len(t) > 2 else 0
        rgb = np.mean([[r["R"], r["G"], r["B"]] for r in t], 0)
        print(f"| {f[0]}-{f[-1]} | {t[0]['t']:.2f}-{t[-1]['t']:.2f} | {x[0]:.0f}->{x[-1]:.0f} | "
              f"{np.mean([r['cy'] for r in t]):.0f} | {max(r['w'] for r in t)}x{max(r['h'] for r in t)} | "
              f"{max(r['peak'] for r in t):.0f} / {np.mean([r['mean'] for r in t]):.0f} | "
              f"({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}) | {vx:+.1f} |")
    # activity over time
    per_sec = {}
    for r in rows:
        s = int(r["t"])
        per_sec[s] = per_sec.get(s, 0) + r["area"]
    print("\nsmudge area (px, summed over content frames) per second:", per_sec)


if __name__ == "__main__":
    main()
