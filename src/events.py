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
    A faint ghost-streak fan: ``n`` thin lines that start at ``x(t)`` at heights
    ``y(t)·f`` (f from 1 down to ``fmin``) and descend linearly to the
    baseline ``len(t)`` px to the right, each with a small wiggle (``wig``
    px, wavelength ``lam``). Keys: ``x, y, len, b``; scalars ``n, fmin, wig,
    lam, th, layer, seed``.

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
        xa, ya, b = _k(ev, "x", t), _k(ev, "y", t), _k(ev, "b", t, 0.05)
        ln = max(_k(ev, "len", t, 300.0), 1.0)
        n = int(ev.get("n", 12))
        fmin = float(ev.get("fmin", 0.2))
        wig, lam = float(ev.get("wig", 6.0)), float(ev.get("lam", 60.0))
        th = float(ev.get("th", 1.0))
        lay = int(ev.get("layer", 1))
        u = np.clip((xs - xa) / ln, 0.0, 1.0)
        ramp = 1.0 - u
        win = b * _window(xs, xa, xa + ln, 12.0)
        r = np.random.default_rng([int(ev.get("seed", 1)), 0x5E7])
        res = []
        for k in range(n):
            f = 1.0 - (1.0 - fmin) * k / max(n - 1, 1)
            ph = r.uniform(0, 2 * np.pi)
            disp = ramp * (ya * f + wig * np.sin(2 * np.pi * xs / lam + ph))
            res.append((disp, win * (0.6 + 0.4 * r.random()), th, lay))
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
        for sm in self.scene.smudges:
            st = np.asarray(sm.t, dtype=np.float64)
            k = keyed(st, sm.duck, t, outside=0.0)
            if k <= 0 or keyed(st, sm.a, t, outside=0.0) <= 0:
                continue
            xc, r = keyed(st, sm.x, t), 2.0 * keyed(st, sm.sx, t)
            w = k * _window(xs, xc - r, xc + r, r)
            D = w if D is None else np.maximum(D, w)
        if D is None:
            return 1.0
        return 1.0 - D[None, :] * np.clip(np.abs(disp) / 8.0, 0.0, 1.0)

    # ------------------------------------------------------------ smudges
    def smudge_buffer(self, t: float, fr) -> np.ndarray:
        """Soft blobs (scene.smudges) in exposure units at output resolution."""
        S = np.zeros((fr.H, fr.W), np.float32)
        for sm in self.scene.smudges:
            if sm.layer < 0:
                self._smudge(S, sm, t, fr)
        return S

    @staticmethod
    def _smudge(buf, sm, t, fr):
        st = np.asarray(sm.t, dtype=np.float64)
        a = keyed(st, sm.a, t, outside=0.0)
        if not a:
            return
        sy = keyed(st, sm.sy, t)
        sy2 = keyed(st, sm.sy2, t) or sy
        add_blob(buf, keyed(st, sm.x, t) * fr.sx, keyed(st, sm.y, t) * fr.sy,
                 keyed(st, sm.sx, t) * fr.sx, sy * fr.sy, sy2 * fr.sy, a,
                 keyed(st, sm.angle, t), sm.pow)

    # ------------------------------------------------------------ raster
    def raster(self, bufs: list, t: float, fr) -> None:
        """Add this frame's event strands into the layer buffers of FrameRenderer `fr`."""
        for sm in self.scene.smudges:
            if sm.layer >= 0:
                self._smudge(bufs[min(sm.layer, len(bufs) - 1)], sm, t, fr)
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


def add_blob(buf: np.ndarray, x: float, y: float, sx: float, sy: float, sy2: float, a: float,
             angle_deg: float = 0.0, p: float = 2.0) -> None:
    """Add exp(-0.5 (|u/sx|^p + |v/s(v)|^p)) * a into buf (output px), where the
    vertical sigma is sy above the centre and sy2 below it."""
    if a == 0 or sx <= 0 or sy <= 0:
        return
    H, W = buf.shape
    r = 3.5 * max(sx, sy, sy2)
    x0, x1 = int(max(0, x - r)), int(min(W, x + r + 1))
    y0, y1 = int(max(0, y - r)), int(min(H, y + r + 1))
    if x1 <= x0 or y1 <= y0:
        return
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5 - x, np.arange(y0, y1) + 0.5 - y)
    th = np.deg2rad(angle_deg)
    u = X * np.cos(th) + Y * np.sin(th)
    v = -X * np.sin(th) + Y * np.cos(th)
    s = np.where(v > 0, sy2, sy)
    buf[y0:y1, x0:x1] += (a * np.exp(-0.5 * (np.abs(u / sx) ** p + np.abs(v / s) ** p))).astype(np.float32)
