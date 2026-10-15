#!/usr/bin/env python3
"""Correlate the source audio loudness with the visual activity.

Decodes 10..23 s of the .mkv audio to memory (mono, 16 kHz float; never written
to disk), computes RMS / band envelopes per video frame and cross-correlates
them with the visual timeline (out/analysis/timeline.csv, run timeline.py first).

Usage: .venv/bin/python tools/analysis/audio.py
"""
import csv
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import N, OUT_DIR, ROOT  # noqa: E402

SR = 16000
T0 = 10.0          # decode start (s), 2 s before the clip for lag analysis


def decode():
    src = sorted(glob.glob(os.path.join(ROOT, "*.mkv")))[0]
    cmd = ["ffmpeg", "-v", "error", "-ss", str(T0), "-i", src, "-t", "13", "-vn", "-ac", "1",
           "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


def band_env(x, lo, hi, centres, win):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f >= hi)] = 0
    y = np.fft.irfft(X, len(x))
    return np.array([np.sqrt(np.mean(y[max(0, c - win):c + win] ** 2) + 1e-12) for c in centres])


def main():
    x = decode()
    fps = 60000 / 1001
    # one centre per video frame from T0 to T0+13 s
    nf = int(13 * fps)
    t = T0 + np.arange(nf) / fps
    centres = ((t - T0) * SR).astype(int)
    win = int(SR / fps)                       # +-1 frame window
    rms = band_env(x, 20, 8000, centres, win)
    voice = band_env(x, 300, 3400, centres, win)
    low = band_env(x, 20, 200, centres, win)
    db = 20 * np.log10(rms + 1e-9)

    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "timeline.csv"))))
    vis = {k: np.array([float(r[k]) for r in rows]) for k in ("mean_luma", "act_frac", "p90_amp")}
    i0 = int(round((12 - T0) * fps))           # audio index of clip frame 0

    print("audio RMS (dBFS, full band) per 0.5 s of the clip:")
    step = int(round(fps / 2))
    print("  " + " ".join(f"{12 + s / fps:.1f}:{db[i0 + s:i0 + s + step].mean():.0f}" for s in range(0, N, step)))
    print("\ncross-correlation (Pearson) of visual vs audio envelopes, lag = audio leads visual by k frames:")
    print("| visual | audio | best lag (frames) | r at best | r at lag 0 |")
    print("|---|---|---|---|---|")
    for vn, v in vis.items():
        vv = (v - v.mean()) / v.std()
        for an, a in (("rms", rms), ("voice 300-3400Hz", voice), ("low <200Hz", low)):
            best = None
            out = {}
            for k in range(-60, 61):
                seg = np.log(a[i0 - k:i0 - k + N] + 1e-6)
                if len(seg) < N:
                    continue
                seg = (seg - seg.mean()) / seg.std()
                r = float(np.mean(vv * seg))
                out[k] = r
                if best is None or r > best[1]:
                    best = (k, r)
            print(f"| {vn} | {an} | {best[0]} | {best[1]:.2f} | {out[0]:.2f} |")
    # smoothed (0.25 s) version: does the slow envelope follow the audio?
    def sm(z, s=7):
        k = np.exp(-0.5 * (np.arange(-3 * s, 3 * s + 1) / s) ** 2)
        return np.convolve(z, k / k.sum(), "same")
    v = sm(vis["act_frac"])
    a = sm(np.log(voice + 1e-6))[i0:i0 + N]
    print(f"\nact_frac vs log voice envelope, both smoothed sigma=7 frames: r = {np.corrcoef(v, a)[0, 1]:.2f}")
    np.savez_compressed(os.path.join(OUT_DIR, "audio_env.npz"), rms=rms[i0:i0 + N], voice=voice[i0:i0 + N],
                        low=low[i0:i0 + N])


if __name__ == "__main__":
    main()
