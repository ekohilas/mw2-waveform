"""Generative waveform model: strands = baseline + ambient wiggle + shared wave packets.

`WaveModel(scene)` expands the procedural strand families and precomputes all
deterministic per-strand / per-(strand, packet) jitters once; `evaluate(t, xs)`
then returns the y position and brightness of every strand at the sample
positions `xs` (design px).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .params.schema import Scene, Strand, keyed

TAU = 2.0 * np.pi


def expand_strands(scene: Scene) -> List[Strand]:
    out = []
    for f, ss in enumerate(scene.strand_sets):
        rng = np.random.default_rng([scene.seed, ss.seed, f, 0xA11])
        for i in range(ss.count):
            y = rng.normal(0.0, ss.y_spread)
            b = ss.brightness * float(np.exp(rng.normal(0.0, ss.brightness_spread)
                                             - 0.5 * ss.brightness_spread ** 2))
            th = ss.thickness * float(np.exp(rng.normal(0.0, ss.thickness_spread)))
            soft = rng.random() < ss.soft_fraction
            layer = ss.layer if ss.layer >= 0 else int(soft)
            ends = rng.random() < ss.end_fraction
            out.append(Strand(y=float(y), brightness=b, thickness=th, layer=layer,
                              seed=1000 * (f + 1) + i, response=ss.response, carrier=ss.carrier,
                              x_end=ss.x_end if ends else 1e9, x_fade=ss.x_fade,
                              knot_amp=ss.knot_amp, knot_lam=ss.knot_lam,
                              knot_speed=ss.knot_speed, vis_amp=ss.vis_amp,
                              vis_floor=ss.vis_floor))
    return out + list(scene.strands)


@dataclass
class StrandEval:
    y: np.ndarray        # [S, N] screen y (design px)
    b: np.ndarray        # [S, N] brightness
    thickness: np.ndarray  # [S]
    layer: np.ndarray    # [S] int


class WaveModel:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.strands = st = expand_strands(scene)
        S = self.S = len(st)
        arr = lambda k: np.array([getattr(s, k) for s in st], dtype=np.float64)
        self.thickness = arr("thickness")
        self.layer = np.array([s.layer for s in st], dtype=int)
        self.base_y = arr("y")
        self.bright = arr("brightness")
        self.response = arr("response")
        self.carrier = arr("carrier")
        self.x_end = arr("x_end")
        self.x_fade = np.maximum(arr("x_fade"), 1e-3)
        self.knot_amp = arr("knot_amp")
        self.vis_amp = arr("vis_amp")
        self.vis_floor = arr("vis_floor")
        self.vis_any = bool((self.vis_amp > 0).any())
        amb = scene.ambient
        n = amb.n
        self.amb_k, self.amb_ph, self.amb_sp, self.amb_a = (np.zeros((S, n)) for _ in range(4))
        self.knot_k, self.knot_ph, self.knot_sp = (np.zeros((S, 2)) for _ in range(3))
        for i, s in enumerate(st):
            r = np.random.default_rng([scene.seed, 0xB22, s.seed])
            lam = np.exp(r.uniform(np.log(amb.lam_min), np.log(amb.lam_max), n))
            self.amb_k[i] = TAU / lam
            self.amb_ph[i] = r.uniform(0, TAU, n)
            self.amb_sp[i] = TAU * r.uniform(-amb.speed, amb.speed, n)
            a = r.normal(0, 1, n)
            self.amb_a[i] = a / max(np.sqrt((a ** 2).sum() / 2), 1e-6)  # unit rms
            self.knot_k[i] = TAU / (s.knot_lam * np.exp(r.normal(0, 0.3, 2)))
            self.knot_ph[i] = r.uniform(0, TAU, 2)
            self.knot_sp[i] = TAU * s.knot_speed * r.uniform(-1, 1, 2)
        self.packets = [(p, np.asarray(p.t, dtype=np.float64)) for p in scene.packets]
        self._pj = {}

    def _jit(self, pi: int):
        """Per-strand jitter for packet index pi: (gain, phase, lam_scale, xshift).

        Seeded by (scene seed, packet index) so appending packets never changes
        the jitter of existing ones.
        """
        if pi not in self._pj:
            j = self.scene.jitter
            r = np.random.default_rng([self.scene.seed, 0xC33, pi])
            z = r.normal(0, 1, (self.S, 4))
            u = r.random(self.S)
            gain = np.exp(j.gain * z[:, 0] - 0.5 * j.gain ** 2)
            gain = np.where(u < self.carrier, gain, j.idle_gain * gain) * self.response
            self._pj[pi] = (gain, j.phase * z[:, 1], 1.0 + j.lam * z[:, 2], j.x * z[:, 3])
        return self._pj[pi]

    def displacement(self, t: float, xs: np.ndarray):
        """Upward displacement [S, N] in design px, and carried envelope |a*gain*env|."""
        sc = self.scene
        disp = np.zeros((self.S, xs.size))
        act = np.zeros((self.S, xs.size)) if self.vis_any else None
        amb = sc.ambient
        ag = amb.amp * keyed(np.asarray(amb.t), amb.gain, t)
        if ag > 0:
            for k in range(amb.n):
                disp += ag * self.amb_a[:, k:k + 1] * np.sin(
                    self.amb_k[:, k:k + 1] * xs[None, :] + self.amb_ph[:, k:k + 1]
                    + self.amb_sp[:, k:k + 1] * t)
        for pi, (p, pt) in enumerate(self.packets):
            if t < pt[0] or t > pt[-1]:
                continue
            a = keyed(pt, p.a, t)
            if a == 0.0:
                continue
            xc = keyed(pt, p.x, t)
            w = max(keyed(pt, p.w, t), 1e-3)
            lam = max(keyed(pt, p.lam, t), 1.0)
            reach = w * (4.0 if p.env_pow <= 2 else 3.0) + 30
            lo, hi = np.searchsorted(xs, [xc - reach, xc + reach])
            if hi <= lo:
                continue
            x = xs[lo:hi][None, :]
            gain, dph, lsc, xsh = self._jit(pi)
            jf = p.jitter
            u = (x - xc - jf * xsh[:, None]) / w
            env = np.exp(-0.5 * np.abs(u) ** p.env_pow)
            arg = TAU * (x - xc) / (lam * (1.0 + jf * (lsc[:, None] - 1.0))) \
                + p.phase + TAU * p.speed * t + jf * dph[:, None]
            c = np.sin(arg)
            if p.harm2:
                c = (c + p.harm2 * np.sin(2 * arg)) / (1.0 + abs(p.harm2))
            if p.sharp != 1.0:
                c = np.sign(c) * np.abs(c) ** p.sharp
            ae = (a * gain)[:, None] * env
            disp[:, lo:hi] += ae * c
            if act is not None:
                act[:, lo:hi] += np.abs(ae)
        return disp, act

    def brightness(self, t: float, xs: np.ndarray, act=None) -> np.ndarray:
        x = xs[None, :]
        knots = 0.5 * (np.sin(self.knot_k[:, :1] * x + self.knot_ph[:, :1] + self.knot_sp[:, :1] * t)
                       + np.sin(self.knot_k[:, 1:] * x + self.knot_ph[:, 1:] + self.knot_sp[:, 1:] * t))
        b = self.bright[:, None] * np.maximum(1.0 + self.knot_amp[:, None] * knots, 0.0)
        end = np.clip((self.x_end[:, None] - x) / self.x_fade[:, None] + 0.5, 0.0, 1.0)
        b = b * end * gain_grid(self.scene, t, xs)[None, :]
        if act is not None:
            va = self.vis_amp[:, None]
            vis = np.where(va > 0, np.clip(act / np.maximum(va, 1e-6), self.vis_floor[:, None], 1.0),
                           1.0)
            b = b * vis
        return b

    def evaluate(self, t: float, xs: np.ndarray) -> StrandEval:
        disp, act = self.displacement(t, xs)
        y = self.scene.baseline_y + self.base_y[:, None] - disp
        return StrandEval(y=y, b=self.brightness(t, xs, act), thickness=self.thickness,
                          layer=self.layer)


def gain_grid(scene: Scene, t: float, xs: np.ndarray) -> np.ndarray:
    g = scene.gain
    tg = np.asarray(g.t, dtype=np.float64)
    G = np.asarray(g.g, dtype=np.float64)
    if len(tg) == 1:
        row = G[0]
    else:
        j = int(np.clip(np.searchsorted(tg, t) - 1, 0, len(tg) - 2))
        f = float(np.clip((t - tg[j]) / max(tg[j + 1] - tg[j], 1e-9), 0, 1))
        row = (1 - f) * G[j] + f * G[j + 1]
    return np.interp(xs, np.asarray(g.x, dtype=np.float64), row)
