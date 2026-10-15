"""Fit-time helpers (these DO read ref/; the renderer never imports them)."""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os

import cv2
import numpy as np

from ..frame import FrameRenderer
from ..params import Scene, scene_from_dict, scene_to_dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_DIR = os.path.join(ROOT, "ref", "frames")
RW, RH = 960, 540
NBIN = 32                      # column bins for coarse maps (30 ref px = 60 design px)
BASE_ROW = 269.5               # baseline row in 960x540


def load_score_module():
    spec = importlib.util.spec_from_file_location("score", os.path.join(ROOT, "tools", "score.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ref_frame(i: int) -> np.ndarray:
    return cv2.imread(os.path.join(REF_DIR, f"{i:04d}.png"))


def measure(img: np.ndarray, thr: float = 40.0):
    """Coarse per-column-bin descriptors of a 960x540 BGR frame.

    extent: rms distance (px) of bright pixels from the baseline
    energy: summed luma
    """
    l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d2 = (np.arange(RH, dtype=np.float32) - BASE_ROW)[:, None] ** 2
    m = (l > thr).astype(np.float32)
    cnt = m.sum(0).reshape(NBIN, -1).sum(1)
    ext = np.sqrt((m * d2).sum(0).reshape(NBIN, -1).sum(1) / np.maximum(cnt, 1))
    en = l.sum(0).reshape(NBIN, -1).sum(1)
    return ext, en


# ------------------------------------------------------------- parallel render
_R = None
_SC = None


def _init(scene_dict):
    global _R, _SC
    cv2.setNumThreads(1)
    _SC = scene_from_dict(scene_dict)
    _R = FrameRenderer(_SC, RW, RH)


def _render_measure(i):
    img = _R.render(i)
    return i, measure(img)


def _render_score(i):
    sm = _render_score.mod
    img = _R.render(i)
    return i, sm.frame_metrics(ref_frame(i), img)


def _score_init(scene_dict):
    _init(scene_dict)
    _render_score.mod = load_score_module()


def pool_map(scene: Scene, fn, idx, init=_init, workers=None):
    with mp.get_context("spawn").Pool(workers or os.cpu_count(), initializer=init,
                                       initargs=(scene_to_dict(scene),)) as pool:
        return dict(pool.imap_unordered(fn, idx, chunksize=4))


def render_measure(scene: Scene, idx):
    return pool_map(scene, _render_measure, idx)


def score_subset(scene: Scene, idx):
    """Mean metrics over frame indices idx (same metrics as tools/score.py)."""
    res = pool_map(scene, _render_score, idx, init=_score_init)
    keys = ["ssim", "ssim_blur", "color", "energy"]
    out = {k: float(np.mean([res[i][k] for i in idx])) for k in keys}
    out["score"] = float(np.mean([out[k] for k in keys]))
    return out
