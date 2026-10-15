"""Generative waveform model: strands = baseline + ambient wiggle + shared wave packets.

`WaveModel(scene)` expands the procedural strand family and precomputes all
deterministic per-strand / per-(strand, packet) jitters once; `evaluate(t, xs)`
then returns the y position and brightness of every strand at the sample
positions `xs` (design px).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .params.schema import Packet, Scene, Strand, keyed

TAU = 2.0 * np.pi


def expand_strands(scene: Scene) -> List[Strand]:
    ss = scene.strand_set
    rng = np.random.default_rng([ss.seed, 0xA11])
    out = []
    for i in range(ss.count):
        y = rng.normal(0.0, ss.y_spread)
        b = ss.brightness * float(np.exp(rng.normal(0.0, ss.brightness_spread)
                                         - 0.5 * ss.brightness_spread ** 2))
        th = ss.thickness * float(np.exp(rng.normal(0.0, ss.thickness_spread)))
        layer = 1 if rng.random() < ss.soft_fraction else 0
        out.append(Strand(y=float(y), brightness=b, thickness=th, layer=layer, seed=i + 1))
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
        self.strands = expand_strands(scene)
        S = len(self.strands)
        self.S = S
        self.thickness = np.array([s.thickness for s in self.strands])
        self.layer = np.array([s.layer for s in self.strands], dtype=int)
        self.base_y = np.array([s.y for s in self.strands])
        self.bright = np.array([s.brightness for s in self.strands])
        seed = scene.strand_set.seed
        amb = scene.ambient
        # ambient sinusoids per strand: [S, n]
        self.amb_k = np.zeros((S, amb.n))
        self.amb_ph = np.zeros((S, amb.n))
        self.amb_sp = np.zeros((S, amb.n))
        self.amb_a = np.zeros((S, amb.n))
        # brightness knots per strand: 2 sinusoids
        self.knot_k = np.zeros((S, 2))
        self.knot_ph = np.zeros((S, 2))
        self.knot_sp = np.zeros((S, 2))
        for i, st in enumerate(self.strands):
            r = np.random.default_rng([seed, 0xB22, st.seed])
            lam = np.exp(r.uniform(np.log(amb.lam_min), np.log(amb.lam_max), amb.n))
            self.amb_k[i] = TAU / lam
            self.amb_ph[i] = r.uniform(0, TAU, amb.n)
            self.amb_sp[i] = TAU * r.uniform(-amb.speed, amb.speed, amb.n)
            a = r.normal(0, 1, amb.n)
            self.amb_a[i] = a / max(np.sqrt((a ** 2).sum() / 2), 1e-6)  # unit rms
            kl = scene.strand_set.knot_lam * np.exp(r.normal(0, 0.3, 2))
            self.knot_k[i] = TAU / kl
            self.knot_ph[i] = r.uniform(0, TAU, 2)
            self.knot_sp[i] = TAU * scene.strand_set.knot_speed * r.uniform(-1, 1, 2)
        self._pj = {}
        self._prep_packets()

    # ------------------------------------------------------------ packets
    def _prep_packets(self):
        self.packets = []
        for pi, p in enumerate(self.scene.packets):
            t = np.asarray(p.t, dtype=np.float64)
            self.packets.append((p, t))

    def _jit(self, pi: int):
        """Per-strand jitter for packet index pi: (gain, phase, lam_scale, xshift)."""
        if pi not in self._pj:
            j = self.scene.jitter
            r = np.random.default_rng([self.scene.strand_set.seed, 0xC33, pi])
            z = r.normal(0, 1, (self.S, 4))
            gain = np.exp(j.gain * z[:, 0] - 0.5 * j.gain ** 2)
            self._pj[pi] = (gain, j.phase * z[:, 1], 1.0 + j.lam * z[:, 2], j.x * z[:, 3])
        return self._pj[pi]

    # ------------------------------------------------------------ evaluation
    def evaluate(self, t: float, xs: np.ndarray) -> StrandEval:
        sc = self.scene
        S = self.S
        N = xs.size
        disp = np.zeros((S, N))  # upward displacement (px)
        # ambient wiggle
        amb = sc.ambient
        ag = amb.amp * keyed(np.asarray(amb.t), amb.gain, t)
        if ag > 0:
            for k in range(amb.n):
                disp += ag * self.amb_a[:, k:k + 1] * np.sin(
                    self.amb_k[:, k:k + 1] * xs[None, :] + self.amb_ph[:, k:k + 1]
                    + self.amb_sp[:, k:k + 1] * t)
        # packets
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
            gain = 1.0 + jf * (gain - 1.0)
            u = (x - xc - jf * xsh[:, None]) / w
            env = np.exp(-0.5 * np.abs(u) ** p.env_pow)
            arg = TAU * (x - xc) / (lam * (1.0 + jf * (lsc[:, None] - 1.0))) \
                + p.phase + TAU * p.speed * t + jf * dph[:, None]
            c = np.sin(arg)
            if p.harm2:
                c = (c + p.harm2 * np.sin(2 * arg)) / (1.0 + abs(p.harm2))
            if p.sharp != 1.0:
                c = np.sign(c) * np.abs(c) ** p.sharp
            disp[:, lo:hi] += (a * gain)[:, None] * env * c
        y = sc.baseline_y + self.base_y[:, None] - disp
        # brightness: per-strand base * knots * gain grid(t, x)
        ka = sc.strand_set.knot_amp
        knots = 0.5 * (np.sin(self.knot_k[:, :1] * xs[None, :] + self.knot_ph[:, :1]
                              + self.knot_sp[:, :1] * t)
                       + np.sin(self.knot_k[:, 1:] * xs[None, :] + self.knot_ph[:, 1:]
                                + self.knot_sp[:, 1:] * t))
        b = self.bright[:, None] * np.maximum(1.0 + ka * knots, 0.0)
        b = b * gain_grid(sc, t, xs)[None, :]
        return StrandEval(y=y, b=b, thickness=self.thickness, layer=self.layer)


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
