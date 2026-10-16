#!/usr/bin/env python3
"""Search for a direct audio-sample -> strand-displacement mapping.

Hypothesis (After Effects "Audio Waveform"-like): the strand displacement in
content frame i is the audio waveform over a window of length D around the
frame time, laid out linearly along x:

    d(x) = A * a_lp( t_i + off + (x - 480) * k ),   k = D / 740 s per px

(a_lp = audio low-passed to the strands' resolution, k < 0 = time runs right
to left, A < 0 = sign flip). For every k on a log grid and every channel
(L, R, mid, side) the normalised cross-correlation (NCC) of each traced strand
(tools/audio/trace.py) against the resampled audio is computed for all
offsets |off| <= OFFMAX via FFT; the best |r| per frame is kept.

The same search against audio taken from elsewhere in the trailer
(t + 40 s: speech/music of the same kind, but the wrong moment) is the null:
the best |r| a search this wide finds by chance.

Output: out/audio/xcorr.npz and a summary table on stdout.
Usage: .venv/bin/python tools/audio/xcorr.py
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X0, X1 = 150, 790                  # columns used (active region)
NX = X1 - X0
XC = 480.0
OFFMAX = 0.4                       # s
DS = np.geomspace(0.005, 3.0, 48)  # window length over 740 px (s)
NULL_SHIFT = float(os.environ.get("NULL_SHIFT", "40"))   # s; env override


def load_audio():
    sr, x = wavfile.read(os.path.join(ROOT, "out", "audio", "full.wav"))
    x = x.astype(np.float64)
    L, R = x[:, 0], x[:, 1]
    return sr, {"L": L, "R": R, "M": (L + R) / 2, "S": (L - R) / 2}


def lowpass(seg, sr, fc):
    F = np.fft.rfft(seg)
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    F *= 1.0 / (1.0 + (f / fc) ** 8)
    return np.fft.irfft(F, len(seg))


def ncc_all(sig, tmpl):
    """NCC of tmpl (len n) against every length-n window of sig."""
    n = len(tmpl)
    t = tmpl - tmpl.mean()
    t /= np.linalg.norm(t) + 1e-12
    num = fftconvolve(sig, t[::-1], mode="valid")
    c1 = np.concatenate([[0], np.cumsum(sig)])
    c2 = np.concatenate([[0], np.cumsum(sig * sig)])
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    var = np.maximum(s2 - s1 * s1 / n, 1e-18)
    return num / np.sqrt(var)


_A = None


def _init():
    global _A
    _A = load_audio()


def frame_job(args):
    j, t, traces, shift = args
    sr, ch = _A
    best = {}
    for D in DS:
        k = D / 740.0
        fc = 1.0 / (2 * 6 * k)                 # keep features >= ~6 px
        ta = t + shift - OFFMAX - (X1 - XC) * k - 0.05
        tb = t + shift + OFFMAX + (XC - X0) * k + 0.05
        i0, i1 = int(ta * sr), int(tb * sr) + 1
        grid = np.arange(ta + 0.025, tb - 0.025, k)
        for cn, a in ch.items():
            seg = lowpass(a[i0:i1], sr, min(fc, sr / 2.2))
            g = np.interp(grid, np.arange(i0, i1) / sr, seg)
            for ti, d in enumerate(traces):
                tm = d[X0:X1]
                if tm.std() < 1:
                    continue
                for mir in (0, 1):
                    r = ncc_all(g, tm[::-1] if mir else tm)
                    # window start time -> offset of window centre wrt t
                    st = grid[: len(r)]
                    if mir:
                        off = st + (X1 - XC) * k - t - shift
                    else:
                        off = st + (XC - X0) * k - t - shift
                    ok = np.abs(off) <= OFFMAX
                    if not ok.any():
                        continue
                    ra = np.abs(r[ok])
                    m = int(np.argmax(ra))
                    key = (cn, mir)
                    val = (ra[m], D, off[ok][m], np.sign(r[ok][m]), ti)
                    if key not in best or val[0] > best[key][0]:
                        best[key] = val
                    kd = ("D", round(D, 6))
                    if kd not in best or ra[m] > best[kd][0]:
                        best[kd] = val
    return j, best


def main():
    z = np.load(os.path.join(ROOT, "out", "audio", "trace.npz"))
    frames, d = z["frames"], z["d"]
    t = 12 + frames * 1001 / 60000
    jobs = []
    for shift in (0.0, NULL_SHIFT):
        jobs += [(j, t[j], d[j], shift) for j in range(len(frames))]
    with ProcessPoolExecutor(initializer=_init) as ex:
        res = list(ex.map(frame_job, jobs, chunksize=4))
    n = len(frames)
    out = {}
    for name, part in (("real", res[:n]), ("null", res[n:])):
        bestr = np.array([max(v[0] for kk, v in b.items() if kk[0] != "D") for _, b in part])
        perD = np.array([[b[("D", round(D, 6))][0] for D in DS] for _, b in part])
        bestD = np.array([max((v for kk, v in b.items() if kk[0] != "D"), key=lambda v: v[0])[1]
                          for _, b in part])
        bestoff = np.array([max((v for kk, v in b.items() if kk[0] != "D"), key=lambda v: v[0])[2]
                            for _, b in part])
        out[name + "_best"] = bestr
        out[name + "_perD"] = perD
        out[name + "_D"] = bestD
        out[name + "_off"] = bestoff
        print(f"{name}: best |r| per frame p10/50/90 = "
              + "/".join(f"{v:.3f}" for v in np.percentile(bestr, [10, 50, 90]))
              + f"  mean {bestr.mean():.3f}")
    print("\nmean best |r| per window length D (real vs null):")
    for q, D in enumerate(DS):
        print(f"  D={D * 1000:8.1f} ms  real {out['real_perD'][:, q].mean():.3f}  "
              f"null {out['null_perD'][:, q].mean():.3f}")
    print("\nfraction of frames with real best > null p95:",
          np.mean(out["real_best"] > np.percentile(out["null_best"], 95)))
    np.savez(os.path.join(ROOT, "out", "audio", f"xcorr_null{NULL_SHIFT:g}.npz"), frames=frames, DS=DS, **out)


if __name__ == "__main__":
    main()
