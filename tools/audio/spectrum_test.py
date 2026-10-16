#!/usr/bin/env python3
"""Test the spectrum hypothesis (AE "Audio Spectrum" / Trapcode Sound Keys):
x = frequency, strand height = band magnitude at the frame time.

Visual: per content frame, per column, the extent of lit light (luma > 40)
above / below the baseline (max |dy|, px), lightly smoothed along x.
Audio: |STFT| (Hann, N samples) at t + lag, laid out along x 100..820 with a
linear or log frequency axis [f_lo, f_hi].

Correlation is computed on per-frame deviations from the clip-mean profile
(both sides), so the stationary hump of the activity region (x ~ 470) cannot
create a spurious match. Score = mean over frames of r (one global mapping).
Null: audio from t + 40 s.

Usage: .venv/bin/python tools/audio/spectrum_test.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from xcorr import NULL_SHIFT, ROOT, load_audio  # noqa: E402

XA, XB = 100, 820
Y0 = 269.5


def visual_extent():
    frames = np.arange(1, 540, 2)
    rows = np.arange(540)[:, None] - Y0
    ext = np.zeros((len(frames), 960))
    for j, i in enumerate(frames):
        l = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "ref", "frames", f"{i:04d}.png")),
                         cv2.COLOR_BGR2GRAY)
        m = l > 40
        e = np.where(m, np.abs(rows), 0).max(0).astype(float)
        ext[j] = cv2.GaussianBlur(e.reshape(1, -1), (0, 0), 2).ravel()
    return frames, ext


def main():
    sr, ch = load_audio()
    a = ch["M"]
    frames, ext = visual_extent()
    V = ext[:, XA:XB]
    V = V - V.mean(0)
    t = 12 + frames * 1001 / 60000
    xs = np.arange(XA, XB)
    u = (xs - XA) / (XB - XA)
    results = []
    for N in (1024, 2048, 4096, 8192):
        win = np.hanning(N)
        f = np.fft.rfftfreq(N, 1 / sr)
        for shift in (0.0, NULL_SHIFT):
            for lag in (-0.1, -0.05, 0.0, 0.05):
                mags = []
                for tt in t:
                    c = int((tt + lag + shift) * sr)
                    mags.append(np.abs(np.fft.rfft(a[c - N // 2:c + N // 2] * win)))
                mags = np.array(mags)
                for axis in ("lin", "log"):
                    for flo in (20, 50, 100, 200):
                        for fhi in (1000, 2000, 4000, 8000, 16000):
                            fx = flo + u * (fhi - flo) if axis == "lin" else flo * (fhi / flo) ** u
                            S = np.array([np.interp(fx, f, m) for m in mags])
                            for comp in ("lin", "db"):
                                Sx = 20 * np.log10(S + 1e-6) if comp == "db" else S
                                Sx = Sx - Sx.mean(0)
                                r = [np.corrcoef(V[j], Sx[j])[0, 1] for j in range(len(t))
                                     if V[j].std() > 0 and Sx[j].std() > 0]
                                results.append((np.mean(r), N, shift, lag, axis, flo, fhi, comp))
    for shift in (0.0, NULL_SHIFT):
        rs = sorted([r for r in results if r[2] == shift], key=lambda r: -abs(r[0]))
        print("REAL" if shift == 0 else "NULL", "top 5 (mean r, N, shift, lag, axis, f_lo, f_hi, compress):")
        for r in rs[:5]:
            print("   ", tuple(round(v, 3) if isinstance(v, float) else v for v in r))


if __name__ == "__main__":
    main()
