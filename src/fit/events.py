"""Fit-time helpers for the scripted events (src/events.py) and smudges.

Scores a scene on the event window (content frames 331..479 by default) with
the exact tools/score.py metrics, keeping a persistent worker pool so a
coordinate search over a few keyframe values is cheap.

    .venv/bin/python -m src.fit.events            # score current scene on the window
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os

import cv2
import numpy as np

from ..frame import FrameRenderer
from ..params import load_scene, scene_from_dict, scene_to_dict
from .common import RW, RH, load_score_module, ref_frame

WINDOW = list(range(331, 480, 2))
KEYS = ["ssim", "ssim_blur", "color", "energy"]

_SM = None
_REF = {}


def _init():
    global _SM
    cv2.setNumThreads(1)
    _SM = load_score_module()


def _job(arg):
    d, i = arg
    sc = scene_from_dict(d)
    img = FrameRenderer(sc, RW, RH).render(i)
    if i not in _REF:
        _REF[i] = ref_frame(i)
    return i, _SM.frame_metrics(_REF[i], img)


class Scorer:
    def __init__(self, frames=None, workers=None):
        self.frames = frames or WINDOW
        self.pool = mp.get_context("spawn").Pool(workers or os.cpu_count(), initializer=_init)

    def per_frame(self, scene):
        d = scene_to_dict(scene) if not isinstance(scene, dict) else scene
        res = dict(self.pool.imap_unordered(_job, [(d, i) for i in self.frames], chunksize=1))
        return {i: float(np.mean([res[i][k] for k in KEYS])) for i in self.frames}, res

    def score(self, scene):
        pf, res = self.per_frame(scene)
        out = {k: float(np.mean([res[i][k] for i in self.frames])) for k in KEYS}
        out["score"] = float(np.mean(list(pf.values())))
        return out

    def close(self):
        self.pool.close()


def tsec(f: float) -> float:
    return round(f * 1001 / 60000, 4)


def coord_search(scorer, scene, params, steps=2, verbose=True):
    """params: list of (getter(scene)->float, setter(scene, v), step). Greedy +-step search."""
    best = scorer.score(scene)["score"]
    for it in range(steps):
        improved = False
        for name, get, put, step in params:
            for sgn in (+1, -1):
                cand = copy.deepcopy(scene)
                put(cand, get(cand) + sgn * step)
                s = scorer.score(cand)["score"]
                if s > best + 1e-5:
                    if verbose:
                        print(f"  {name}: {get(scene):.3f} -> {get(cand):.3f}  {best:.4f} -> {s:.4f}")
                    best, scene, improved = s, cand, True
                    break
        if not improved:
            break
    return scene, best


if __name__ == "__main__":
    sc = Scorer()
    print(sc.score(load_scene()))
    sc.close()
