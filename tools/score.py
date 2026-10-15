#!/usr/bin/env python3
"""Score a rendered clip against the reference frames in ref/frames.

Usage: .venv/bin/python tools/score.py <render.mp4 | frames_dir> [--json out.json]

Both sides are compared at 960x540 (the reference source is only 720p).
Every metric is normalised so an all-black render scores ~0 and an exact
match scores 1; the mostly-black background earns no credit.
Per-frame metrics:
  ssim      - mean SSIM map (luma) over the lit region (either frame's
              luma > 8, dilated 7px): fine structure and line placement
  ssim_blur - the same after a sigma=6px Gaussian blur: glow and layout,
              tolerant of small line offsets
  color     - 1 - sum|dRGB| / sum(max(RGB)) on sigma=6px blurred frames
  energy    - 1 - sum|dE| / sum(max(E)), E = column-wise brightness profile
              (how much "signal" is at each x position)
The overall score is the mean of the four metrics, averaged over frames.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity

W, H = 960, 540
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "ref", "frames")


def load_frames(path):
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.endswith(".png"))
        for n in names:
            img = cv2.imread(os.path.join(path, n))
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            yield img
        return
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    proc.wait()


def luma(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def frame_metrics(ref, out):
    ly_r, ly_o = luma(ref), luma(out)
    lit = cv2.dilate(((ly_r > 8) | (ly_o > 8)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if not lit.any():
        return dict(ssim=1.0, ssim_blur=1.0, color=1.0, energy=1.0)
    _, smap = structural_similarity(ly_r, ly_o, data_range=255, full=True)
    ssim = float(smap[lit].mean())
    br = cv2.GaussianBlur(ref, (0, 0), 6).astype(np.float32)
    bo = cv2.GaussianBlur(out, (0, 0), 6).astype(np.float32)
    lb_r = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
    lb_o = cv2.cvtColor(bo, cv2.COLOR_BGR2GRAY)
    _, smap = structural_similarity(lb_r, lb_o, data_range=255, full=True)
    ssim_blur = float(smap[lit].mean())
    color = 1.0 - float(np.abs(br - bo).sum() / max(np.maximum(br, bo).sum(), 1e-6))
    e_r = ly_r.astype(np.float32).sum(0)
    e_o = ly_o.astype(np.float32).sum(0)
    e_r = cv2.GaussianBlur(e_r.reshape(1, -1), (0, 0), 8).ravel()
    e_o = cv2.GaussianBlur(e_o.reshape(1, -1), (0, 0), 8).ravel()
    energy = 1.0 - float(np.abs(e_r - e_o).sum() / max(np.maximum(e_r, e_o).sum(), 1e-6))
    return dict(ssim=ssim, ssim_blur=ssim_blur, color=color, energy=energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render")
    ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ref_names = sorted(n for n in os.listdir(REF_DIR) if n.endswith(".png"))
    rows = []
    for i, out in enumerate(load_frames(args.render)):
        if i >= len(ref_names):
            break
        ref = cv2.imread(os.path.join(REF_DIR, ref_names[i]))
        m = frame_metrics(ref, out)
        m["score"] = float(np.mean(list(m.values())))
        rows.append(m)
    if len(rows) != len(ref_names):
        print(f"WARNING: render has {len(rows)} frames, reference has {len(ref_names)}; "
              "missing frames score 0", file=sys.stderr)
    n = len(ref_names)
    keys = ["ssim", "ssim_blur", "color", "energy", "score"]
    summary = {k: float(sum(r[k] for r in rows) / n) for k in keys}
    if not args.quiet:
        # Per-second breakdown helps find which part of the clip is weakest.
        fps = 60000 / 1001
        for s in range(0, 9):
            seg = rows[int(s * fps):int((s + 1) * fps)]
            if seg:
                print(f"  t={12 + s:>2}-{13 + s}s  " + "  ".join(
                    f"{k}={np.mean([r[k] for r in seg]):.3f}" for k in keys))
    print("TOTAL  " + "  ".join(f"{k}={summary[k]:.4f}" for k in keys))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "frames": rows}, f)


if __name__ == "__main__":
    main()
