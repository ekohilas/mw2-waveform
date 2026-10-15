#!/usr/bin/env python3
"""Per-frame timeline of the reference: energy, x centroid/spread, amplitude
envelope, cuts/flashes, frame holds, and wave-packet tracks.

Usage: .venv/bin/python tools/analysis/timeline.py [--bin 16]
Writes (gitignored):
  out/analysis/timeline.csv   one row per frame
  out/analysis/envelope.npz   up/dn amplitude + activity per x-bin per frame
                              (fitting targets for the renderer's envelopes)
  out/analysis/packets.csv    packet tracks
Prints the summary tables used in docs/analysis.md.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import FPS, H, N, OUT_DIR, W, load_all_luma, src_time  # noqa: E402

Y0 = 269.5          # baseline centre (px, 960x540)
CORE = 8            # |y-Y0| <= CORE counts as baseline, not "activity"
LIT = 40            # luma threshold for amplitude extents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=16)
    args = ap.parse_args()
    L = load_all_luma()
    ys = np.arange(H, dtype=np.float32) - Y0
    off = (np.abs(ys) > CORE)[:, None]
    xs = np.arange(W, dtype=np.float32)
    nb = W // args.bin

    rows, up_b, dn_b, act_b = [], [], [], []
    prev = None
    for i in range(N):
        l = L[i]
        colE = l.sum(0)
        act = np.where(off, l, 0).sum(0)
        lit = l > LIT
        up = np.clip(np.where(lit, -ys[:, None], -1).max(0), 0, None)
        dn = np.clip(np.where(lit, ys[:, None], -1).max(0), 0, None)
        tot = colE.sum()
        cx = (colE * xs).sum() / tot
        sx = np.sqrt((colE * (xs - cx) ** 2).sum() / tot)
        at = act.sum() + 1e-6
        ax = (act * xs).sum() / at
        asx = np.sqrt((act * (xs - ax) ** 2).sum() / at)
        diff = 0.0 if prev is None else float(np.abs(l - prev).mean())
        prev = l
        rows.append(dict(frame=i, t=round(src_time(i), 4), mean_luma=round(float(l.mean()), 3),
                         act_frac=round(float(act.sum() / tot), 3), cx=round(float(cx), 1),
                         sx=round(float(sx), 1), act_cx=round(float(ax), 1), act_sx=round(float(asx), 1),
                         max_up=float(up.max()), max_dn=float(dn.max()),
                         p90_amp=float(np.percentile(np.maximum(up, dn), 90)),
                         diff_prev=round(diff, 3)))
        up_b.append(up[:nb * args.bin].reshape(nb, -1).max(1))
        dn_b.append(dn[:nb * args.bin].reshape(nb, -1).max(1))
        act_b.append(act[:nb * args.bin].reshape(nb, -1).sum(1))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "timeline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    up_b, dn_b, act_b = map(np.array, (up_b, dn_b, act_b))
    np.savez_compressed(os.path.join(OUT_DIR, "envelope.npz"), up=up_b, dn=dn_b, act=act_b,
                        bin=args.bin, y0=Y0)

    # ---- holds / cuts / flashes
    d = np.array([r["diff_prev"] for r in rows])
    holds = np.where(d[1:] < 0.1)[0] + 1
    print(f"held frames (diff<0.1 vs previous): {len(holds)}; odd={np.sum(holds % 2 == 1)} even={np.sum(holds % 2 == 0)}")
    new = np.where(d >= 0.1)[0]
    print("content-change diffs (mean |dL|) pct 1/50/99/max:",
          np.percentile(d[new], [1, 50, 99, 100]).round(2))
    ml = np.array([r["mean_luma"] for r in rows])
    jump = np.abs(np.diff(ml))
    print("largest |d mean_luma| between frames:", np.sort(jump)[-5:].round(2), "at", np.argsort(jump)[-5:] + 1)

    # ---- per-half-second summary
    print("\n| t (s) | frames | mean luma | activity frac | act x-centroid | act x-spread | p90 amp | max up | max dn |")
    print("|---|---|---|---|---|---|---|---|---|")
    step = int(round(FPS / 2))
    for s in range(0, N, step):
        seg = rows[s:s + step]
        g = lambda k, fn=np.mean: fn([r[k] for r in seg])  # noqa: E731
        print(f"| {seg[0]['t']:.1f} | {s}-{s + len(seg) - 1} | {g('mean_luma'):.2f} | {g('act_frac'):.2f} | "
              f"{g('act_cx'):.0f} | {g('act_sx'):.0f} | {g('p90_amp'):.0f} | {g('max_up', np.max):.0f} | {g('max_dn', np.max):.0f} |")

    # ---- packet tracks: segments of smoothed amplitude > thr in each content frame,
    # linked across frames by x-overlap (allow a gap of 2 frames).
    amp = np.maximum(up_b, dn_b).astype(np.float32)
    amp = cv2.GaussianBlur(amp, (0, 0), sigmaX=1.0, sigmaY=0.01)
    thr = 25
    tracks, active = [], []
    for i in range(N):
        on = amp[i] > thr
        segs = []
        j = 0
        while j < nb:
            if on[j]:
                k = j
                while k + 1 < nb and on[k + 1]:
                    k += 1
                segs.append((j, k))
                j = k + 1
            else:
                j += 1
        still = []
        used = set()
        for tr in active:
            a0, a1 = tr["cur"]
            hit = [s for s in segs if s[0] <= a1 + 1 and s[1] >= a0 - 1]
            if hit:
                lo = min(s[0] for s in hit)
                hi = max(s[1] for s in hit)
                for s in hit:
                    used.add(s)
                tr["cur"] = (lo, hi)
                tr["end"] = i
                tr["x0"] = min(tr["x0"], lo)
                tr["x1"] = max(tr["x1"], hi)
                tr["peak"] = max(tr["peak"], float(amp[i, lo:hi + 1].max()))
                tr["path"].append((i, (lo + hi + 1) / 2))
                still.append(tr)
            elif i - tr["end"] <= 2:
                still.append(tr)
            else:
                tracks.append(tr)
        for s in segs:
            if s not in used:
                still.append(dict(start=i, end=i, cur=s, x0=s[0], x1=s[1],
                                  peak=float(amp[i, s[0]:s[1] + 1].max()), path=[(i, (s[0] + s[1] + 1) / 2)]))
        # merge tracks whose current spans overlap
        merged = []
        for tr in sorted(still, key=lambda t: t["cur"][0]):
            if merged and tr["cur"][0] <= merged[-1]["cur"][1] and tr["end"] == i and merged[-1]["end"] == i:
                m = merged[-1]
                m["cur"] = (m["cur"][0], max(m["cur"][1], tr["cur"][1]))
                m["start"] = min(m["start"], tr["start"])
                m["x0"] = min(m["x0"], tr["x0"])
                m["x1"] = max(m["x1"], tr["x1"])
                m["peak"] = max(m["peak"], tr["peak"])
            else:
                merged.append(tr)
        active = merged
    tracks += active
    tracks = [t for t in tracks if t["end"] - t["start"] >= 6]
    tracks.sort(key=lambda t: t["start"])
    b = args.bin
    with open(os.path.join(OUT_DIR, "packets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "t_start", "t_end", "x0", "x1", "peak_amp_px", "drift_px_per_frame"])
        print(f"\n| # | frames | t (s) | x range (px) | peak amp (px) | centre drift (px/frame) |")
        print("|---|---|---|---|---|---|")
        for n, t in enumerate(tracks):
            p = np.array(t["path"])
            drift = np.polyfit(p[:, 0], p[:, 1] * b, 1)[0] if len(p) > 3 else 0.0
            row = [t["start"], t["end"], round(src_time(t["start"]), 2), round(src_time(t["end"]), 2),
                   t["x0"] * b, (t["x1"] + 1) * b, round(t["peak"]), round(drift, 2)]
            w.writerow(row)
            print(f"| {n} | {row[0]}-{row[1]} | {row[2]:.2f}-{row[3]:.2f} | {row[4]}-{row[5]} | {row[6]} | {row[7]:+.2f} |")


if __name__ == "__main__":
    main()
