"""Per-frame renderer: waveform -> layered cores -> smudges -> glow -> colour."""
from __future__ import annotations

import numpy as np

from .color import ramp_lut, to_bgr8
from .params.schema import Scene, keyed
from .raster import add_gaussian_blob, blur, raster_strands
from .waveform import WaveModel


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

    # -------------------------------------------------------------- layers
    def strand_layers(self, t: float) -> list:
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
                sel = ev.layer == li if force_layer is None else (
                    np.ones_like(ev.layer, bool) if li == force_layer else
                    np.zeros_like(ev.layer, bool))
                if sel.any():
                    bufs[li] += raster_strands(y[sel], b[sel], th[sel], self.xs_out,
                                               self.W, self.H)
        return bufs

    def linear(self, i: int) -> np.ndarray:
        sc = self.scene
        t = sc.frame_time(i)
        bufs = self.strand_layers(t)
        L = np.zeros((self.H, self.W), np.float32)
        for lay, buf in zip(sc.layers, bufs):
            L += lay.gain * blur(buf, lay.blur * self.sy)
        for sm in sc.smudges:
            st = np.asarray(sm.t, dtype=np.float64)
            a = keyed(st, sm.a, t, outside=0.0)
            if a:
                add_gaussian_blob(L, keyed(st, sm.x, t) * self.sx, keyed(st, sm.y, t) * self.sy,
                                  keyed(st, sm.sx, t) * self.sx, keyed(st, sm.sy, t) * self.sy,
                                  a, keyed(st, sm.angle, t))
        out = L.copy()
        for g in sc.look.glow:
            out += g.weight * blur(L, g.sigma * self.sy)
        return out

    def render(self, i: int) -> np.ndarray:
        """BGR uint8 [H, W, 3]."""
        return to_bgr8(self.linear(i), self.scene.look, self.lut)
