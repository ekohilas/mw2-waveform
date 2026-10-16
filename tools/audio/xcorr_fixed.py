#!/usr/bin/env python3
"""Global (frame-consistent) test of the audio-waveform hypothesis.

A real After-Effects-style mapping uses ONE window length D, ONE offset, ONE
sign and ONE direction for every frame. So instead of the best |r| per frame
(xcorr.py, which a wide search inflates), this averages the signed NCC
r_i(D, off) over all content frames and looks for a peak:

    R(D, off) = mean_i r_i(D, off)

A true mapping gives a sharp peak with R close to the per-frame r (>0.5);
no mapping gives |R| ~ a few / sqrt(270). Signals tested: the first DP trace
(trace.npz d[:,0]) and the signed off-baseline centroid (cen). Channels L, R,
mid, side; both x directions. Null: the same with audio from t + 40 s.

Output: out/audio/xcorr_fixed.npz and a table on stdout.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from xcorr import DS, NULL_SHIFT, OFFMAX, ROOT, X0, X1, XC, load_audio, lowpass, ncc_all

_A = None


def _init():
    global _A
    _A = load_audio()


def job(args):
    D, sig_name, shift = args
    sr, ch = _A
    z = np.load(os.path.join(ROOT, "out", "audio", "trace.npz"))
    frames = z["frames"]
    S = z["d"][:, 0] if sig_name == "trace" else z["cen"]
    t_all = 12 + frames * 1001 / 60000
    k = D / 740.0
    step = max(k, 0.0002)
    offs = np.arange(-OFFMAX, OFFMAX + step / 2, step)
    res = {}
    for cn, a in ch.items():
        fc = min(1.0 / (12 * k), sr / 2.2)
        # low-pass the whole clip once
        i0, i1 = int((11.0 + shift) * sr), int((22.5 + shift) * sr)
        seg = lowpass(a[i0:i1], sr, fc)
        tt = np.arange(i0, i1) / sr
        for mir in (0, 1):
            acc = np.zeros(len(offs))
            n = 0
            for j, t in enumerate(t_all):
                tm = S[j, X0:X1]
                if tm.std() < 1:
                    continue
                tm = tm[::-1] if mir else tm
                ta = t + shift - OFFMAX - (X1 - XC) * k - 0.01
                tb = t + shift + OFFMAX + (XC - X0) * k + 0.01
                grid = np.arange(ta, tb, k)
                g = np.interp(grid, tt, seg)
                r = ncc_all(g, tm)
                st = grid[: len(r)]
                off = st + ((X1 - XC) if mir else (XC - X0)) * k - t - shift
                acc += np.interp(offs, off, r)
                n += 1
            R = acc / n
            m = int(np.argmax(np.abs(R)))
            res[(cn, mir)] = (R[m], offs[m], float(np.sqrt(np.mean(R ** 2))))
    return D, sig_name, shift, res


def main():
    jobs = [(D, s, sh) for D in DS for s in ("trace", "cen") for sh in (0.0, NULL_SHIFT)]
    with ProcessPoolExecutor(initializer=_init) as ex:
        out = list(ex.map(job, jobs))
    table = {}
    for D, s, sh, res in out:
        best = max(res.items(), key=lambda kv: abs(kv[1][0]))
        table[(D, s, sh)] = best
    print("best |mean r| over (channel, direction, offset) per window length D")
    print(f"{'D ms':>8} | {'trace real':>22} | {'trace null':>10} | {'cen real':>22} | {'cen null':>10}")
    rows = []
    for D in DS:
        tr, tn = table[(D, "trace", 0.0)], table[(D, "trace", NULL_SHIFT)]
        cr, cnl = table[(D, "cen", 0.0)], table[(D, "cen", NULL_SHIFT)]
        print(f"{D * 1000:8.1f} | {tr[1][0]:+.3f} {tr[0][0]}{'m' if tr[0][1] else ' '} off {tr[1][1] * 1000:+6.1f} | "
              f"{abs(tn[1][0]):10.3f} | {cr[1][0]:+.3f} {cr[0][0]}{'m' if cr[0][1] else ' '} off {cr[1][1] * 1000:+6.1f} | "
              f"{abs(cnl[1][0]):10.3f}")
        rows.append([D, tr[1][0], tr[1][1], abs(tn[1][0]), cr[1][0], cr[1][1], abs(cnl[1][0])])
    np.save(os.path.join(ROOT, "out", "audio", "xcorr_fixed.npy"), np.array(rows))


if __name__ == "__main__":
    main()
