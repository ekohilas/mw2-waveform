"""Scripted events: compact, keyframed objects drawn on top of the waveform.

The analyst's moving features (docs/analysis.md §3.4, §4) are not wave
packets: they are a handful of coherent objects with their own paths. Each
event is one dict in ``scene.events`` with a ``type`` and a few keyframes
(``t`` in seconds, lengths in design px at 1920x1080, positive heights = up).
Keyed values follow the usual convention: scalar = constant, list = aligned
with ``t`` and linearly interpolated. Every event is invisible outside
``[t[0], t[-1]]``.

Types
-----
``needle_pair``
    Two ECG-like needles born at ``x0`` and moving apart: needle centre
    ``x0 ± d(t)``. Each needle is one stroke along the baseline that shoots up
    ``up(t)`` px to a sharp tip, crosses down to ``dn(t)`` px below the
    baseline ``gap`` px further *outward*, and returns to the baseline after
    another ``w`` px. Keys: ``d, up, dn, b`` (brightness); scalars ``x0, w,
    gap, th`` (core thickness), ``layer``.

``bump``
    A cluster of ``n`` tall rounded spikes that travels along ``x(t)`` (the
    left-travelling burst). Strand k sits at ``x(t) + spread·u_k`` with height
    ``a(t)·(1 + jit·z_k)`` and a flat-topped profile
    ``exp(-½|dx/w|^p)``; the offsets/heights are re-drawn every
    ``redraw`` s from ``seed`` (the strands redraw like the rest of the
    waveform while the cluster's envelope moves coherently). A strand points
    down with probability ``down``. Keys: ``x, a, b``; scalars ``n, w,
    spread, jit, p, th, layer, seed, redraw, down``.

``streaks``
    Faint ghost streaks: ``n`` thin parallel diagonals trailing an anchor
    ``x(t), y(t)`` (height above baseline) down to the baseline, ``len`` px to
    the right. Keys: ``x, y, b``; scalars ``n, len, spacing, th, layer``.

Smudges (soft glows) stay in ``scene.smudges`` (`frame.FrameRenderer`).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .params.schema import keyed
from .raster import raster_strands


def _k(ev: dict, name: str, t: float, default=0.0) -> float:
    return keyed(np.asarray(ev["t"], dtype=np.float64), ev.get(name, default), t)


def _window(xs: np.ndarray, lo: float, hi: float, taper: float = 6.0) -> np.ndarray:
    """1 inside [lo, hi], linear taper to 0 over `taper` px outside."""
    return np.clip(np.minimum(xs - lo, hi - xs) / max(taper, 1e-3) + 1.0, 0.0, 1.0)


class EventModel:
    """Evaluates ``scene.events``; returns extra strands per frame."""

    def __init__(self, scene):
        self.scene = scene
        self.events: List[dict] = [e for e in getattr(scene, "events", []) or []
                                   if e.get("on", True)]

    def active(self, t: float) -> List[dict]:
        return [e for e in self.events if e["t"][0] <= t <= e["t"][-1]]

    # ------------------------------------------------------------ strands
    def strands(self, t: float, xs: np.ndarray):
        """List of (disp [N] px up, b [N], thickness, layer) at design xs."""
        out = []
        for ev in self.active(t):
            fn = getattr(self, "_" + ev["type"])
            out.extend(fn(ev, t, xs))
        return out

    def _needle_pair(self, ev, t, xs):
        x0 = float(ev["x0"])
        d = _k(ev, "d", t)
        up, dn, b = _k(ev, "up", t), _k(ev, "dn", t), _k(ev, "b", t, 1.0)
        w = float(ev.get("w", 10.0))
        gap = float(ev.get("gap", 16.0))
        th = float(ev.get("th", 1.2))
        lay = int(ev.get("layer", 0))
        res = []
        for s in (-1.0, 1.0):
            xc = x0 + s * d
            # control points along the outward direction s
            px = np.array([xc - s * w, xc, xc + s * gap, xc + s * (gap + w)])
            py = np.array([0.0, up, -dn, 0.0])
            if s < 0:
                px, py = px[::-1], py[::-1]
            disp = np.interp(xs, px, py, left=0.0, right=0.0)
            bb = b * _window(xs, px[0], px[-1])
            res.append((disp, bb, th, lay))
        return res

    def _bump(self, ev, t, xs):
        xc = _k(ev, "x", t)
        a, b = _k(ev, "a", t), _k(ev, "b", t, 1.0)
        n = int(ev.get("n", 2))
        w = float(ev.get("w", 16.0))
        spread = float(ev.get("spread", 30.0))
        jit = float(ev.get("jit", 0.3))
        p = float(ev.get("p", 3.0))
        th = float(ev.get("th", 1.4))
        lay = int(ev.get("layer", 0))
        down = float(ev.get("down", 0.0))
        redraw = float(ev.get("redraw", 0.0667))
        c = int(np.floor(t / redraw + 1e-6))
        r = np.random.default_rng([int(ev.get("seed", 1)), 0xE77, c])
        res = []
        for k in range(n):
            u, z, v, q = r.normal(), r.normal(), r.random(), r.normal()
            xk = xc + spread * u
            ak = a * max(1.0 + jit * z, 0.1) * (-1.0 if v < down else 1.0)
            wk = w * float(np.exp(0.25 * q))
            reach = 3.0 * wk
            lo, hi = np.searchsorted(xs, [xk - reach, xk + reach])
            if hi <= lo:
                continue
            disp = np.zeros_like(xs)
            disp[lo:hi] = ak * np.exp(-0.5 * np.abs((xs[lo:hi] - xk) / wk) ** p)
            res.append((disp, b * _window(xs, xk - reach, xk + reach), th, lay))
        return res

    def _streaks(self, ev, t, xs):
        xa, ya, b = _k(ev, "x", t), _k(ev, "y", t), _k(ev, "b", t, 0.2)
        n = int(ev.get("n", 6))
        ln = float(ev.get("len", 200.0))
        sp = float(ev.get("spacing", 8.0))
        th = float(ev.get("th", 1.0))
        lay = int(ev.get("layer", 1))
        res = []
        for k in range(n):
            x0 = xa + k * sp
            disp = np.interp(xs, [x0, x0 + ln], [ya - k * sp * 0.5, 0.0], left=0.0, right=0.0)
            res.append((disp, b * _window(xs, x0, x0 + ln, 10.0), th, lay))
        return res

    # ------------------------------------------------------------ ducking
    def _regions(self, ev, t):
        """(lo, hi) x ranges this event occupies (design px)."""
        ty = ev["type"]
        if ty == "needle_pair":
            x0, d = float(ev["x0"]), _k(ev, "d", t)
            r = float(ev.get("w", 10.0)) + float(ev.get("gap", 16.0))
            return [(x0 - d - r, x0 - d + r), (x0 + d - r, x0 + d + r)]
        if ty == "bump":
            xc, sp = _k(ev, "x", t), float(ev.get("spread", 30.0))
            r = 1.5 * sp + 2.0 * float(ev.get("w", 16.0))
            return [(xc - r, xc + r)]
        return []

    def duck(self, t: float, xs: np.ndarray, disp: np.ndarray) -> np.ndarray:
        """Brightness factor [S, N] for the waveform strands: inside an event's
        footprint the displaced (packet) parts of the waveform are dimmed by
        the event's ``duck`` (0..1, keyable) so the event replaces, rather
        than adds to, the local activity. `disp` = upward displacement [S, N]."""
        D = None
        for ev in self.active(t):
            k = _k(ev, "duck", t)
            if k <= 0:
                continue
            m = float(ev.get("duck_margin", 20.0))
            for lo, hi in self._regions(ev, t):
                w = k * _window(xs, lo - m, hi + m, 2.0 * m)
                D = w if D is None else np.maximum(D, w)
        if D is None:
            return 1.0
        return 1.0 - D[None, :] * np.clip(np.abs(disp) / 8.0, 0.0, 1.0)

    # ------------------------------------------------------------ raster
    def raster(self, bufs: list, t: float, fr) -> None:
        """Add this frame's event strands into the layer buffers of FrameRenderer `fr`."""
        if not self.events:
            return
        st = self.strands(t, fr.xs)
        if not st:
            return
        base = self.scene.baseline_y
        for li in range(len(bufs)):
            sel = [s for s in st if min(s[3], len(bufs) - 1) == li]
            if not sel:
                continue
            y = np.stack([(base - s[0]) * fr.sy for s in sel])
            b = np.stack([s[1] for s in sel])
            th = np.array([s[2] for s in sel]) * fr.sy
            bufs[li] += raster_strands(y, b, th, fr.xs_out, fr.W, fr.H)
