#!/usr/bin/env python3
"""Static look of the reference: baseline position, line cross-section,
colour ramp, core-vs-glow tint, background level, horizontal fade.

Usage: .venv/bin/python tools/analysis/static_look.py
Prints the tables used in docs/analysis.md (all units: px at 960x540, 8-bit).
"""
import os
import sys

import cv2
import numpy as np
from scipy.optimize import curve_fit, least_squares

sys.path.insert(0, os.path.dirname(__file__))
from common import H, N, W, load, load_all_luma, luma  # noqa: E402

R = 40                      # half-length of the cross-section window
BAND = slice(255, 286)      # rows searched for the baseline peak
SIG = [1.0, 3.3, 7.6]       # 3-Gaussian decomposition of the cross-section


def baseline(L):
    rp = L.mean((0, 2))
    w = np.clip(rp - 1, 0, None)
    am = np.argmax(L[:, BAND, :], 1) + BAND.start
    print(f"baseline: mean row-profile peak row {rp.argmax()}, centroid {np.sum(w * np.arange(H)) / w.sum():.2f}")
    print("median per-column peak row at x=0,80,..,880:", np.median(am, 0)[::80].astype(int).tolist())
    print("row profile (mean luma) above/below baseline at |dy| = 30,60,100,150:",
          [(round(float(rp[270 - d]), 2), round(float(rp[270 + d]), 2)) for d in (30, 60, 100, 150)])


def quiet_profiles(L):
    """Vertical luma/RGB profiles through the baseline at columns with no wave activity."""
    profs, rgbp, pks = [], [], []
    for i in range(0, N, 5):
        l = L[i]
        outside = l.sum(0) - l[BAND].sum(0)
        rgb = load(i)
        for x in range(20, W - 20, 3):
            if outside[x] >= 40:
                continue
            c = l[:, x]
            y = int(np.argmax(c[BAND])) + BAND.start
            if c[y] < 60:
                continue
            a, b, cc = c[y - 1], c[y], c[y + 1]
            d = (a - cc) / (2 * (a - 2 * b + cc) + 1e-9)       # sub-pixel peak
            yy = y + d + np.arange(-R, R + 1)
            profs.append(np.interp(yy, np.arange(H), c))
            rgbp.append(np.stack([np.interp(yy, np.arange(H), rgb[:, x, k]) for k in range(3)], 1))
            pks.append(c[y])
    return np.array(profs), np.array(rgbp), np.array(pks)


def fit_profile(profs, pks):
    x = np.arange(-R, R + 1).astype(float)
    p = np.median(profs, 0)
    p = (p + p[::-1]) / 2
    half = p >= p[R] / 2
    print(f"\nquiet-column samples: {len(profs)}; median-profile FWHM {half.sum()} px; "
          f"per-sample FWHM p25/50/75 =",
          np.percentile([(pr >= pr[R] / 2).sum() for pr in profs], [25, 50, 75]).tolist())
    print("median luma profile at |dy|=0..15:", [round(float(p[R + o]), 1) for o in range(16)])

    def g3(x, a1, s1, a2, s2, a3, s3, c):
        return np.maximum(0, a1 * np.exp(-x**2 / (2 * s1**2)) + a2 * np.exp(-x**2 / (2 * s2**2))
                          + a3 * np.exp(-x**2 / (2 * s3**2)) - c)
    po, _ = curve_fit(g3, x, p, p0=[100, 0.8, 80, 2.5, 30, 7, 3], maxfev=50000)
    print("3-Gaussian fit (a,sigma)x3, clip:", po.round(2).tolist(),
          "rms", round(float(np.sqrt(((g3(x, *po) - p) ** 2).mean())), 2))

    def ge(x, a1, s1, a2, lam, c):
        return np.maximum(0, a1 * np.exp(-x**2 / (2 * s1**2)) + a2 * np.exp(-np.abs(x) / lam) - c)
    po, _ = curve_fit(ge, x, p, p0=[150, 1, 80, 3, 3], maxfev=20000)
    print("Gauss+exp fit a1,s1,a2,lambda,clip:", po.round(2).tolist(),
          "rms", round(float(np.sqrt(((ge(x, *po) - p) ** 2).mean())), 2))
    for lo, hi in [(60, 120), (120, 200), (200, 256)]:
        s = (pks >= lo) & (pks < hi)
        q = np.median(profs[s], 0)
        print(f"  peak {lo}-{hi} (n={s.sum()}): |dy|=0,1,2,3,4,6,8,10,12,15 ->",
              [round(float((q[R - o] + q[R + o]) / 2), 1) for o in [0, 1, 2, 3, 4, 6, 8, 10, 12, 15]])


def fit_colour_profile(rgbp, pks):
    """Per-channel exposure model: out_c = 255*(1-exp(-sum_k A_kc*G_k(y))) - o."""
    y = np.arange(-R, R + 1).astype(float)
    basis = np.stack([np.exp(-y**2 / (2 * s * s)) for s in SIG], 1)

    def tm(E, o):
        return np.maximum(0, 255 * (1 - np.exp(-E)) - o)
    print("\ncolour cross-section model out_c = 255*(1-exp(-E_c)) - o,  E_c = sum_k A_kc exp(-y^2/2s_k^2)")
    for name, sel in [("bright (peak>=200)", pks >= 200), ("dim (peak 80-160)", (pks >= 80) & (pks < 160))]:
        P = np.median(rgbp[sel], 0)
        P = (P + P[::-1]) / 2

        def res(p):
            A = p[:9].reshape(3, 3)
            return np.concatenate([tm(basis @ A[:, c], p[9]) - P[:, c] for c in range(3)])
        r = least_squares(res, np.r_[[1, 1, .5, .5, .5, .2, .1, .2, .02], 3], bounds=(0, [20] * 9 + [30]))
        A = r.x[:9].reshape(3, 3)
        print(f"  {name}: offset o={r.x[9]:.1f}, rms={np.sqrt((r.fun**2).mean()):.2f}")
        for k, s in enumerate(SIG):
            print(f"    sigma {s}: A_R={A[k, 0]:.3f} A_G={A[k, 1]:.3f} A_B={A[k, 2]:.3f}")


def colour_ramp():
    px = []
    for i in range(0, N, 10):
        rgb = load(i).reshape(-1, 3)
        lu = luma(rgb)
        m = lu > 2
        px.append(np.c_[lu[m], rgb[m]])
    px = np.concatenate(px)
    bins = [2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 235, 245, 256]
    print("\n| luma | median R | G | B | R/G p10-p50-p90 |")
    print("|---|---|---|---|---|")
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = px[(px[:, 0] >= lo) & (px[:, 0] < hi)]
        med = np.median(s[:, 1:], 0)
        rg = s[:, 1] / np.maximum(s[:, 2], 1)
        print(f"| {lo}-{hi} | {med[0]:.0f} | {med[1]:.0f} | {med[2]:.0f} | "
              f"{np.percentile(rg, 10):.2f}-{np.median(rg):.2f}-{np.percentile(rg, 90):.2f} |")
    # core vs halo tint at equal luma
    acc = []
    for i in range(1, N, 12):
        rgb = load(i)
        lu = luma(rgb)
        mx = cv2.dilate(lu, np.ones((7, 7), np.uint8))
        m = (lu >= 70) & (lu < 110)
        acc.append(np.c_[mx[m], rgb[m]])
    a = np.concatenate(acc)
    print("\npixels with luma 70-110, split by 7x7 neighbourhood max (dim core vs halo of bright core):")
    for lo, hi in [(70, 110), (110, 160), (160, 220), (220, 256)]:
        s = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
        print(f"  nbr max {lo}-{hi}: R/G={np.median(s[:, 1] / s[:, 2]):.2f} B/G={np.median(s[:, 3] / s[:, 2]):.2f}")


def background_and_fade(L):
    far = np.r_[0:100, 440:540]
    v = L[:, far, :]
    print(f"\nbackground (rows 0-99, 440-539): mean {v.mean():.3f}, frac>0.5 {(v > 0.5).mean():.4f}")
    E = L.sum(1)
    pk = L[:, BAND, :].max(1)
    print("\n| x | median column energy | median baseline peak luma |")
    print("|---|---|---|")
    for x in [0, 2, 5, 10, 40, 80, 120, 160, 240, 320, 400, 480, 560, 640, 720, 780, 800, 810, 820, 830,
              840, 880, 920, 950, 959]:
        print(f"| {x} | {np.median(E[:, x]):.0f} | {np.median(pk[:, x]):.0f} |")
    bb = np.array([cv2.GaussianBlur(L[i], (0, 0), 1)[262:278].max(0) for i in range(1, N, 4)])
    print("baseline brightness (x 100-800) p5/25/50/75/95:", np.percentile(bb[:, 100:800], [5, 25, 50, 75, 95]).round(0))
    ac = []
    for b in bb:
        s = b[100:800] - b[100:800].mean()
        f = np.fft.irfft(np.abs(np.fft.rfft(s, 2048)) ** 2)[:60]
        ac.append(f / f[0])
    print("baseline brightness autocorr, lags 0..40 step 4:", np.mean(ac, 0)[:41:4].round(2).tolist())


def main():
    L = load_all_luma()
    baseline(L)
    profs, rgbp, pks = quiet_profiles(L)
    fit_profile(profs, pks)
    fit_colour_profile(rgbp, pks)
    colour_ramp()
    background_and_fade(L)


if __name__ == "__main__":
    main()
