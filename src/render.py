"""Entry point.

    .venv/bin/python -m src.render                 # full 1920x1080 -> out/render.mp4
    .venv/bin/python -m src.render --preview       # 960x540 PNGs -> out/preview/
    .venv/bin/python -m src.render --preview --score
    .venv/bin/python -m src.render --still 360     # one PNG -> out/still_0360.png

Only the unique images are rendered (the source animates at 29.97 fps with
every image held for two output frames, see Scene.hold); they are rendered in
parallel and written in order. Never reads ref/ or any source media.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2

from .encode import Mp4Writer, PngDirWriter
from .frame import FrameRenderer
from .params import load_scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_R = None


def _init(scene_path, W, H):
    global _R
    cv2.setNumThreads(1)
    _R = FrameRenderer(load_scene(scene_path), W, H)


def _work(i):
    return i, _R.render(i)


def render_all(scene_path=None, out=None, preview=False, workers=None, crf=16, verbose=True):
    scene = load_scene(scene_path)
    W, H = (960, 540) if preview else (scene.width, scene.height)
    if out is None:
        out = os.path.join(ROOT, "out", "preview" if preview else "render.mp4")
    uniq = sorted({scene.source_index(i) for i in range(scene.n_frames)})
    writer = (PngDirWriter(out) if preview or not out.endswith(".mp4")
              else Mp4Writer(out, W, H, scene.fps_num, scene.fps_den, crf=crf))
    t0 = time.time()
    workers = workers or os.cpu_count()
    done = {}
    nxt = 0
    with mp.get_context("spawn").Pool(workers, initializer=_init,
                                       initargs=(scene_path, W, H)) as pool:
        for i, img in pool.imap_unordered(_work, uniq, chunksize=2):
            done[i] = img
            # flush every output frame whose source image is ready
            while nxt < scene.n_frames and scene.source_index(nxt) in done:
                writer.write(done[scene.source_index(nxt)])
                nxt += 1
                if nxt < scene.n_frames:
                    keep = scene.source_index(nxt)
                    for k in [k for k in done if k < keep]:
                        del done[k]
            if verbose and len(done) and (i % 60 == 1):
                print(f"\r  {nxt}/{scene.n_frames} frames  {time.time() - t0:.1f}s",
                      end="", file=sys.stderr)
    writer.close()
    dt = time.time() - t0
    if verbose:
        print(f"\rrendered {len(uniq)} unique images ({scene.n_frames} frames) at {W}x{H} "
              f"in {dt:.1f}s = {dt / len(uniq) * workers:.2f} s/image/core -> {out}",
              file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="960x540 PNG frames for quick scoring")
    ap.add_argument("--out", help="output .mp4 or directory")
    ap.add_argument("--scene", help="scene JSON (default src/params/scene.json)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--still", type=int, help="render a single frame index to PNG")
    ap.add_argument("--score", action="store_true", help="run tools/score.py on the output")
    args = ap.parse_args()
    if args.still is not None:
        scene = load_scene(args.scene)
        W, H = (960, 540) if args.preview else (scene.width, scene.height)
        out = args.out or os.path.join(ROOT, "out", f"still_{args.still:04d}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.time()
        cv2.imwrite(out, FrameRenderer(scene, W, H).render(args.still))
        print(f"{out} ({time.time() - t0:.2f}s)")
        return
    out = render_all(args.scene, args.out, args.preview, args.workers, args.crf)
    if args.score:
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "score.py"), out], check=True)


if __name__ == "__main__":
    main()
