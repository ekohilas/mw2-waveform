"""Per-frame renderer: waveform -> layered cores -> glow -> colour (+ smudges)."""
from __future__ import annotations

import numpy as np

from .color import ramp_lut, to_bgr8
from .params.schema import Scene, keyed
from .raster import add_gaussian_blob, blur, raster_strands
from .waveform import WaveModel

SQRT_2PI = float(np.sqrt(2 * np.pi))


class FrameRenderer:
    """Renders frames of `scene` at W x H. Construct once per process."""

    def __init__(self, scene: Scene, W: int = None, H: int = None):
        self.scene = scene
        self.W = W or scene.width
        self.H = H or scene.height
        self.sx = self.W / 1920.0
        self.sy = self.H / 1080.0
        self.model = WaveModel(scene)
        self.lut = ramp_lut(scene.look)
        n = int(round(self.W * scene.supersample))
        self.xs_out = (np.arange(n + 1) / scene.supersample).astype(np.float64)
        self.xs = self.xs_out / self.sx  # design px

    # -------------------------------------------------------------- stages
    def strand_layers(self, t: float) -> list:
        """One float32 core buffer per scene layer (before the layer blur)."""
        sc = self.scene
        bufs = [np.zeros((self.H, self.W), np.float32) for _ in sc.layers]
        passes = [(t, 0.0, 0.0, 1.0, None)]
        e = sc.echo
        for k in range(1, e.n + 1):
            passes.append((t - k * e.dt, k * e.dx, k * e.dy, e.decay ** k, e.layer))
        for (tt, ox, oy, g, force_layer) in passes:
            ev = self.model.evaluate(tt, self.xs - ox)
            y = (ev.y + oy) * self.sy
            b = ev.b * g
            th = ev.thickness * self.sy
            for li in range(len(sc.layers)):
                if force_layer is None:
                    sel = ev.layer == li
                else:
                    sel = np.full(ev.layer.shape, li == force_layer)
                if sel.any():
                    bufs[li] += raster_strands(y[sel], b[sel], th[sel], self.xs_out,
                                               self.W, self.H)
        return bufs

    def core(self, t: float) -> np.ndarray:
        L = np.zeros((self.H, self.W), np.float32)
        for lay, buf in zip(self.scene.layers, self.strand_layers(t)):
            L += lay.gain * blur(buf, lay.blur * self.sy)
        return L

    def smudge_buffer(self, t: float) -> np.ndarray:
        S = np.zeros((self.H, self.W), np.float32)
        for sm in self.scene.smudges:
            st = np.asarray(sm.t, dtype=np.float64)
            a = keyed(st, sm.a, t, outside=0.0)
            if a:
                add_gaussian_blob(S, keyed(st, sm.x, t) * self.sx, keyed(st, sm.y, t) * self.sy,
                                  keyed(st, sm.sx, t) * self.sx, keyed(st, sm.sy, t) * self.sy,
                                  a, keyed(st, sm.angle, t))
        return S

    def exposure(self, i: int) -> np.ndarray:
        """Per-channel linear exposure E [H, W, 3] (BGR), channels mode."""
        sc = self.scene
        t = sc.frame_time(i)
        L = self.core(t) * sc.look.exposure
        E = np.zeros((self.H, self.W, 3), np.float32)
        for comp in sc.look.components:
            # peak-normalised: a line of brightness b, thickness th (design px)
            # gets E_peak ~= rgb * b * th
            g = blur(L, comp.sigma * self.sy) * (comp.sigma * SQRT_2PI if comp.sigma > 0.05 else 1.0 / self.sy)
            for c in range(3):
                if comp.rgb[2 - c]:
                    E[:, :, c] += comp.rgb[2 - c] * g
        if sc.smudges:
            S = self.smudge_buffer(t)
            for c in range(3):
                E[:, :, c] += sc.look.smudge_rgb[2 - c] * S
        return E

    def linear(self, i: int) -> np.ndarray:
        """Single-channel light (ramp mode): core + smudges + glow."""
        sc = self.scene
        t = sc.frame_time(i)
        L = self.core(t) + self.smudge_buffer(t)
        out = L.copy()
        for g in sc.look.glow:
            out += g.weight * blur(L, g.sigma * self.sy)
        return out

    def render(self, i: int) -> np.ndarray:
        """BGR uint8 [H, W, 3]."""
        look = self.scene.look
        if look.mode == "channels":
            E = self.exposure(i)
            v = 1.0 - np.exp(-np.maximum(E, 0.0)) - look.offset
            return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return to_bgr8(self.linear(i), look, self.lut)
