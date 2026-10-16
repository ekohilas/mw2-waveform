#!/usr/bin/env python3
"""Audio around the scripted events of docs/analysis.md §3.4.

Prints the 20-ms log-RMS envelope (dBFS, mid channel), high-band (> 2 kHz)
envelope and a spectral-flux onset strength per content frame, for the needle
pair (birth at frame 339) and the left-travelling burst (frames 357-475).
Also tests whether the burst path x(f) could be a transient sliding through a
fixed-length window (implied window length D = 740 px / speed).

Usage: .venv/bin/python tools/audio/events.py
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
from xcorr import load_audio  # noqa: E402


def env_db(sig, sr, t, win=0.02):
    c = (t * sr).astype(int)
    h = int(win * sr / 2)
    return np.array([10 * np.log10(np.mean(sig[i - h:i + h] ** 2) + 1e-12) for i in c])


def flux(sig, sr, t, n=1024):
    w = np.hanning(n)
    out = []
    for tt in t:
        c = int(tt * sr)
        a = np.abs(np.fft.rfft(sig[c - n:c] * w))
        b = np.abs(np.fft.rfft(sig[c - 2 * n:c - n] * w))
        out.append(np.sum(np.maximum(np.log1p(a * 100) - np.log1p(b * 100), 0)))
    return np.array(out)


def main():
    sr, ch = load_audio()
    m = ch["M"]
    hi = sosfiltfilt(butter(4, 2000, "highpass", fs=sr, output="sos"), m)
    fr = np.arange(321, 480, 2)
    t = 12 + fr * 1001 / 60000
    e, eh, fl = env_db(m, sr, t), env_db(hi, sr, t), flux(m, sr, t)
    print("frame  t(s)   env dB  >2k dB  flux")
    for f, tt, a, b, c in zip(fr, t, e, eh, fl):
        mark = " <- needle pair birth" if f == 339 else (" <- burst start" if f == 357 else "")
        print(f"{f:5d} {tt:7.3f} {a:7.1f} {b:7.1f} {c:6.1f}{mark}")
    fz = (fl - fl.mean()) / fl.std()
    top = fr[np.argsort(fz)[-6:][::-1]]
    print("strongest onsets (frames):", list(top))
    # burst path from analysis §3.4
    bf = np.array([360, 387, 405, 423, 441, 459, 471])
    bx = np.array([620, 520, 460, 383, 263, 117, 70])
    v = np.polyfit(bf, bx, 1)[0] * 60000 / 1001   # px/s (960 scale)
    print(f"burst speed {v:.0f} px/s -> a sliding window would need D = 740/|v| = {740 / abs(v):.2f} s;"
          f" the needle pair moves +-5.8 px/frame = {5.8 * 60000 / 1001:.0f} px/s (D = "
          f"{740 / (5.8 * 60000 / 1001):.2f} s) but in BOTH directions at once")


if __name__ == "__main__":
    main()
