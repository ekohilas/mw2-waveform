#!/usr/bin/env python3
"""What partial signal does the audio give? Envelope, lag, x-dependence, onsets.

Visual side: out/analysis/envelope.npz (tools/analysis/timeline.py): per output
frame and 16-px bin, `up`/`dn` amplitude and `act` off-baseline energy; only
the 270 content frames (odd) are used. Columns 6..51 (x 96..831).

1. Global: per-frame activity vs audio log-RMS for bands x windows x lags.
2. Per x-bin best lag: a sliding (scrolling) envelope would show lag linear in x.
3. Onsets: audio onset times (rises of the log envelope) vs visual onsets.

Writes out/audio/envelope_prior.npz: per content frame the fitted audio-
predicted activity and amplitude (a prior for inverse rendering).
Usage: .venv/bin/python tools/audio/envelope.py
"""
import os
import sys

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
from xcorr import ROOT, load_audio  # noqa: E402

FR = np.arange(1, 540, 2)
T = 12 + FR * 1001 / 60000
BANDS = {"full": None, "<300": (20, 300), "300-3k": (300, 3000), "1k-4k": (1000, 4000),
         ">4k": (4000, 20000)}
WINS = (0.01, 0.02, 0.033, 0.05, 0.1, 0.2)
LAGS = np.round(np.arange(-0.2, 0.2001, 0.01), 3)


def log_env(sig, sr, t, win, lag):
    c = ((t + lag) * sr).astype(int)
    h = int(win * sr / 2)
    cs = np.concatenate([[0], np.cumsum(sig.astype(np.float64) ** 2)])
    return 0.5 * np.log((cs[c + h] - cs[c - h]) / (2 * h) + 1e-12)


def main():
    sr, ch = load_audio()
    a = ch["M"]
    z = np.load(os.path.join(ROOT, "out", "analysis", "envelope.npz"))
    act = z["act"][FR].astype(float)[:, 6:52]
    amp = np.maximum(z["up"], z["dn"])[FR].astype(float)[:, 6:52]
    vis = {"log act": np.log1p(act.sum(1)), "mean amp": amp.mean(1),
           "p90 amp": np.percentile(amp, 90, axis=1)}
    filt = {bn: (a if b is None else sosfiltfilt(butter(4, b, "bandpass", fs=sr, output="sos"), a))
            for bn, b in BANDS.items()}

    print("1. global envelope: best Pearson r over window/lag (lag > 0: audio later than frame)")
    print(f"{'band':>7} | " + " | ".join(f"{vn:>26}" for vn in vis))
    best_all = None
    for bn, s in filt.items():
        cells = []
        for vn, v in vis.items():
            b = max((np.corrcoef(v, log_env(s, sr, T, w, l))[0, 1], w, l) for w in WINS for l in LAGS)
            cells.append(f"r={b[0]:.3f} w={b[1] * 1000:3.0f}ms lag={b[2] * 1000:+4.0f}")
            if vn == "log act" and (best_all is None or b[0] > best_all[0]):
                best_all = (b[0], bn, b[1], b[2])
        print(f"{bn:>7} | " + " | ".join(cells))
    r0, bn, w, l = best_all
    print(f"best for log act: band {bn}, window {w * 1000:.0f} ms, lag {l * 1000:+.0f} ms, r = {r0:.3f}")

    # lag curve at the best band/window
    s = filt[bn]
    print("   lag curve:", " ".join(f"{lg * 1000:+.0f}:{np.corrcoef(vis['log act'], log_env(s, sr, T, w, lg))[0, 1]:.2f}"
                                  for lg in LAGS[::2]))

    print("\n2. per x-bin best lag (log act per bin vs audio env); sliding envelope => lag linear in x")
    e_by_lag = {lg: log_env(s, sr, T, w, lg) for lg in LAGS}
    lag_x, r_x = [], []
    for b in range(act.shape[1]):
        v = np.log1p(act[:, b])
        rs = [(np.corrcoef(v, e_by_lag[lg])[0, 1], lg) for lg in LAGS]
        r, lg = max(rs)
        lag_x.append(lg)
        r_x.append(r)
    xs = (np.arange(6, 52) * 16 + 8)
    for b in range(0, len(xs), 3):
        print(f"   x={xs[b]:4d}  best lag {lag_x[b] * 1000:+5.0f} ms  r={r_x[b]:.2f}")
    slope = np.polyfit(xs, lag_x, 1)[0]
    print(f"   linear fit of lag vs x: {slope * 1e6:.1f} us/px  (scroll speed {1 / slope if slope else np.inf:.0f} px/s)")

    print("\n3. onsets")
    ev = log_env(s, sr, T, w, l)
    va = vis["log act"]
    def onsets(x):
        d = np.diff(x, prepend=x[0])
        p, _ = find_peaks(d, height=np.percentile(d, 85), distance=4)
        return p
    oa, ov = onsets(ev), onsets(va)
    for tol in (1, 2, 3):
        hit = np.mean([np.min(np.abs(ov - p)) <= tol for p in oa])
        rnd = []
        rng = np.random.default_rng(0)
        for _ in range(200):
            q = np.sort(rng.choice(len(T), len(ov), replace=False))
            rnd.append(np.mean([np.min(np.abs(q - p)) <= tol for p in oa]))
        print(f"   audio onsets matched by a visual onset within +-{tol} content frames: "
              f"{hit:.2f} (random {np.mean(rnd):.2f})  [{len(oa)} audio, {len(ov)} visual onsets]")

    # prior: linear fit act/amp ~ audio env
    A = np.vstack([ev, np.ones_like(ev)]).T
    ca, *_ = np.linalg.lstsq(A, va, rcond=None)
    cp, *_ = np.linalg.lstsq(A, vis["p90 amp"], rcond=None)
    print(f"\nprior fits: log act = {ca[0]:.3f}*env + {ca[1]:.2f};  p90 amp = {cp[0]:.2f}*env + {cp[1]:.1f}")
    np.savez(os.path.join(ROOT, "out", "audio", "envelope_prior.npz"), frames=FR, t=T, env=ev,
             pred_log_act=A @ ca, pred_p90_amp=A @ cp, band=bn, win=w, lag=l, r=r0)


if __name__ == "__main__":
    main()
