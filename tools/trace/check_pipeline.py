"""Does what the fitter optimises at 960x540 survive the real delivery path?

    .venv/bin/python -m tools.trace.check_pipeline 101 251 345 [--traces out/traces]

For each traced content frame: render at 1920x1080 (src.traces), x264-encode
exactly like the full render (src.encode.Mp4Writer, with and without the level
compensation), decode + area-downscale like tools/score.py, and compare the
score metrics with the direct 960x540-equivalent render (2x supersampled +
2x2 average, i.e. what the fitter sees).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import score  # noqa: E402
from src import traces as T  # noqa: E402
from src.encode import Mp4Writer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=int, nargs="+")
    ap.add_argument("--traces", default=T.TRACE_DIR)
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()
    dev = T.default_device()
    cv = T.Canvas(2, device=dev)
    look_t = T.look_tensors(T.load_look(args.traces), dev)
    imgs = {c: T.render_trace(c, cv, look_t, args.traces) for c in args.frames}
    res = {}
    for comp in (False, True):
        path = os.path.join(ROOT, "out", "scratch", f"check_{int(comp)}.mp4")
        w = Mp4Writer(path, cv.W, cv.H, 60000, 1001, crf=args.crf, compensate=comp)
        for c in args.frames:
            for _ in range(2):
                w.write(imgs[c])
        w.close()
        dec = list(score.load_frames(path))
        res[comp] = [dec[2 * k] for k in range(len(args.frames))]
    keys = ["ssim", "ssim_blur", "color", "energy"]
    print("frame  " + "  ".join(f"{k:>24}" for k in keys) + "   (direct / x264 raw / x264 compensated)")
    for k_, c in enumerate(args.frames):
        ref = cv2.imread(os.path.join(ROOT, "ref", "frames", f"{c:04d}.png"))
        direct = cv2.resize(imgs[c], (960, 540), interpolation=cv2.INTER_AREA)
        ms = [score.frame_metrics(ref, im) for im in (direct, res[False][k_], res[True][k_])]
        print(f"c{c:04d} " + "  ".join("%.4f/%.4f/%.4f" % tuple(m[k] for m in ms) for k in keys))


if __name__ == "__main__":
    main()
