"""Coarse first-pass fit: a fixed grid of wave packets + a brightness grid.

Keeps the global look of the base scene (strands, glow, ramp, ...) and
replaces `packets` and `gain` with:
  * one packet per 60-px column bin (32 packets), fixed x / width / carrier,
    amplitude keyed at ~15 Hz,
  * a (time x column) brightness grid at the same keys,
fitted by iteratively matching coarse per-bin descriptors (rms vertical extent
of lit pixels, summed luma) of the render against the reference.

    .venv/bin/python -m src.fit.coarse [--iters 6] [--scene IN] [--out OUT]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..params import GainGrid, Packet, load_scene, save_scene
from .common import NBIN, ROOT, measure, ref_frame, render_measure, score_subset


def key_frames(scene):
    ks = list(range(scene.hold_offset, scene.n_frames, 2 * scene.hold))
    last = scene.source_index(scene.n_frames - 1)
    if ks[-1] != last:
        ks.append(last)
    return ks


def main():
    args = parse_args()
    sc = fit(load_scene(args.scene), args)
    save_scene(sc, args.out)
    sub = list(range(1, sc.n_frames, 10))
    print("subset score", score_subset(sc, sub))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lam", type=float, nargs=2, default=[80.0, 220.0])
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--w", type=float, default=45.0)
    ap.add_argument("--amp_gain", type=float, default=2.0)
    return ap.parse_args(argv)


def fit(sc, args, verbose=True):
    ks = key_frames(sc)
    tk = [round(sc.frame_time(k), 4) for k in ks]
    cache = os.path.join(ROOT, "out", "fitcache_coarse.npz")
    if os.path.exists(cache) and len(np.load(cache)["ext"]) == len(ks):
        z = np.load(cache)
        ext_ref, en_ref = z["ext"], z["en"]
    else:
        m = [measure(ref_frame(k)) for k in ks]
        ext_ref = np.array([e for e, _ in m])
        en_ref = np.array([e for _, e in m])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, ext=ext_ref, en=en_ref)

    rng = np.random.default_rng(args.seed)
    bw = 1920.0 / NBIN
    xc = (np.arange(NBIN) + 0.5) * bw
    A = np.maximum(ext_ref - 4.0, 0.0) * 2.8          # [K, NBIN] design px
    G = np.ones_like(A)
    lam = np.exp(rng.uniform(np.log(args.lam[0]), np.log(args.lam[1]), NBIN))
    speed = rng.uniform(-args.speed, args.speed, NBIN)
    phase = rng.uniform(0, 2 * np.pi, NBIN)

    def apply():
        sc.packets = [Packet(t=tk, x=float(xc[j]), w=args.w, a=[float(v) for v in A[:, j]],
                             lam=float(lam[j]), phase=float(phase[j]), speed=float(speed[j]),
                             id=f"grid{j:02d}") for j in range(NBIN)]
        sc.gain = GainGrid(t=tk, x=[float(v) for v in xc], g=G.round(3).tolist())

    for it in range(args.iters + 1):
        apply()
        res = render_measure(sc, ks)
        ext_r = np.array([res[k][0] for k in ks])
        en_r = np.array([res[k][1] for k in ks])
        e_err = np.abs(ext_r - ext_ref).mean()
        n_err = np.abs(en_r - en_ref).sum() / np.maximum(en_r, en_ref).sum()
        if verbose:
            print(f"iter {it}: extent mae {e_err:.2f} px, energy rel err {n_err:.3f}")
        if it == args.iters:
            break
        A = np.clip(A + args.amp_gain * (ext_ref - ext_r), 0.0, 520.0)
        G = np.clip(G * ((en_ref + 2e3) / (en_r + 2e3)) ** 0.8, 0.02, 20.0)
    return sc


if __name__ == "__main__":
    main()
