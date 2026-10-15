"""Scene description: plain dataclasses that round-trip to/from JSON.

Units
-----
All spatial quantities are in *design pixels* of a 1920x1080 frame
(x right, y down). The renderer scales them to the output resolution.
All times are in seconds relative to the start of the section
(frame i is at t = i * 1001 / 60000, so t runs 0 .. ~9.0).

Keyed values
------------
Anything that animates is stored as keyframes: a time array ``t`` plus value
arrays of the same length (or a scalar, meaning constant). Values are linearly
interpolated between keys. This is deliberately simple so a fitter can treat
every key as an independent, bounded parameter.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np

Num = Union[float, List[float]]


def keyed(t: np.ndarray, v: Num, tq: float, outside: Optional[float] = None) -> float:
    """Evaluate a keyed value (scalar or array aligned with ``t``) at time ``tq``.

    Outside the key range the end value is held, unless ``outside`` is given.
    """
    if np.isscalar(v):
        return float(v)
    v = np.asarray(v, dtype=np.float64)
    if outside is not None and (tq < t[0] or tq > t[-1]):
        return outside
    return float(np.interp(tq, t, v))


# --------------------------------------------------------------------------
# Waveform
# --------------------------------------------------------------------------
@dataclass
class Packet:
    """A wave packet: a Gaussian-enveloped carrier that every strand shares.

    Displacement contributed to strand s at (x, t):
        g_s * a(t) * env((x - x(t)) / w(t)) * shape(sin(2*pi*(x - x(t)) / (lam*l_s)
                                                  + phase + 2*pi*speed*t + d_s))
    where g_s, l_s, d_s are deterministic per-(strand, packet) jitters.
    ``a`` is in design pixels (positive = upwards on screen).
    ``a`` is zero outside [t[0], t[-1]].
    """
    t: List[float]
    x: Num = 960.0        # envelope centre (px)
    w: Num = 80.0         # envelope sigma (px)
    a: Num = 0.0          # amplitude (px, peak displacement)
    lam: Num = 120.0      # carrier wavelength (px)
    phase: float = 0.0    # carrier phase at t=0 (rad)
    speed: float = 0.0    # carrier phase speed (cycles/s); travelling wave
    sharp: float = 1.0    # shape exponent p: sign(c)*|c|**p (p>1 = spikier peaks)
    harm2: float = 0.0    # 2nd harmonic ratio (asymmetry)
    env_pow: float = 2.0  # envelope exponent (2 = Gaussian, >2 = flatter top)
    jitter: float = 1.0   # scales the strand jitter for this packet
    id: str = ""


@dataclass
class StrandJitter:
    """How strongly each strand deviates from the shared packet shape."""
    gain: float = 0.35        # lognormal sigma of per-strand gain
    phase: float = 0.6        # rad, stddev of per-strand phase offset
    lam: float = 0.08         # relative stddev of wavelength
    x: float = 6.0            # px, stddev of per-strand envelope shift
    # Per (strand, packet): with probability Strand.carrier the strand carries
    # the packet at full gain, otherwise it only follows it at `idle_gain`.
    idle_gain: float = 0.15


@dataclass
class Ambient:
    """Low-level idle wiggle on every strand (always present)."""
    amp: float = 3.0              # px rms per strand
    lam_min: float = 60.0
    lam_max: float = 400.0
    speed: float = 0.4            # cycles/s (max)
    n: int = 5                    # sinusoids per strand
    # amplitude multiplier keyed over time
    t: List[float] = field(default_factory=lambda: [0.0])
    gain: Num = 1.0


@dataclass
class StrandSet:
    """Procedural family of strands; expanded deterministically from ``seed``."""
    count: int = 24
    seed: int = 1
    y_spread: float = 2.0          # px stddev of baseline offset
    brightness: float = 1.0        # mean core brightness
    brightness_spread: float = 0.6 # lognormal sigma
    thickness: float = 1.6         # px (design) full core thickness
    thickness_spread: float = 0.4  # lognormal sigma
    soft_fraction: float = 0.35    # fraction of strands drawn on the soft layer
    knot_amp: float = 0.6          # brightness modulation along x (0..1)
    knot_lam: float = 150.0        # px wavelength of brightness knots
    knot_speed: float = 0.3        # cycles/s
    # how strongly these strands follow the wave packets
    response: float = 1.0          # gain multiplier on all packets
    carrier: float = 1.0           # P(strand fully carries a given packet), else idle_gain
    # horizontal extent: a strand ends at x_end (soft over x_fade px); with
    # probability end_fraction, otherwise it spans the whole frame
    x_end: float = 1920.0
    x_fade: float = 50.0
    end_fraction: float = 0.0
    layer: int = -1                # force layer (-1: soft_fraction decides)
    # visibility follows activity: if vis_amp > 0 the strand's brightness is
    # scaled by clip(local packet amplitude carried by this strand / vis_amp,
    # vis_floor, 1), so packet strands vanish into nothing when idle
    vis_amp: float = 0.0
    vis_floor: float = 0.0


@dataclass
class Strand:
    """An explicit strand (normally produced by expanding a StrandSet)."""
    y: float = 0.0
    brightness: float = 1.0
    thickness: float = 1.6
    layer: int = 0
    seed: int = 0
    response: float = 1.0
    carrier: float = 1.0
    x_end: float = 1e9
    x_fade: float = 50.0
    knot_amp: float = 0.6
    knot_lam: float = 150.0
    knot_speed: float = 0.3
    vis_amp: float = 0.0
    vis_floor: float = 0.0


@dataclass
class Layer:
    """Core buffer with its own pre-blur (sharp lines vs soft/out-of-focus)."""
    blur: float = 0.0     # design px sigma applied to this layer's core
    gain: float = 1.0


@dataclass
class Glow:
    sigma: float          # design px
    weight: float


@dataclass
class Echo:
    """Ghost copies of the waveform at earlier times (phosphor/ribbon trail)."""
    n: int = 0
    dt: float = 0.05      # s between echoes
    decay: float = 0.5    # brightness factor per echo
    dx: float = 0.0       # px offset per echo
    dy: float = 0.0
    layer: int = 1


@dataclass
class GainGrid:
    """Bilinear brightness multiplier over (t, x). g has shape [len(t)][len(x)]."""
    t: List[float] = field(default_factory=lambda: [0.0])
    x: List[float] = field(default_factory=lambda: [0.0, 1920.0])
    g: List[List[float]] = field(default_factory=lambda: [[1.0, 1.0]])


@dataclass
class Smudge:
    """Soft anisotropic Gaussian blob added to the light buffer."""
    t: List[float]
    x: Num = 960.0
    y: Num = 540.0
    sx: Num = 60.0
    sy: Num = 30.0
    a: Num = 0.0          # peak intensity (linear units), 0 outside key range
    angle: Num = 0.0      # degrees
    duck: Num = 0.0       # dim displaced waveform strands under the blob (src/events.py)
    sy2: Num = 0.0        # sigma below the centre (0: = sy), for glows hanging to the baseline
    pow: float = 2.0      # profile exponent (2 = Gaussian, >2 = flat-topped)
    # -1: exposure-space blob tinted look.smudge_rgb; >=0: light added to that
    # layer's core buffer (gets the lines' colour/glow pipeline; a ~0.1-0.3)
    layer: int = -1


@dataclass
class Component:
    """One term of the per-channel exposure model (mode "channels").

    E_c += rgb[c] * (peak-normalised Gaussian of the core buffer, sigma px).
    """
    sigma: float          # design px
    rgb: List[float]      # [A_R, A_G, A_B]
    # optional per-channel response exponent: E_c += rgb[c] * g**gamma[c]
    # (gamma < 1 on R/B makes dim lines paler than bright ones)
    gamma: Optional[List[float]] = None


@dataclass
class Look:
    # "ramp": v = tonemap(I + glow) -> colour ramp lookup
    # "channels": E_c = sum_k rgb_k[c] * G_sigma_k(I); out_c = 1 - exp(-E_c) - offset
    mode: str = "ramp"
    components: List[Component] = field(default_factory=list)
    offset: float = 0.0   # subtracted after the tonemap (channels mode), 0..1
    smudge_rgb: List[float] = field(default_factory=lambda: [0.3, 1.0, 0.1])
    exposure: float = 1.0
    # colour ramp: stops [v, r, g, b] with v = tonemapped intensity in 0..1,
    # rgb in 0..1 (display-referred, sRGB-ish)
    ramp: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0]])
    glow: List[Glow] = field(default_factory=list)
    tonemap: str = "exp"   # "exp": v = 1 - exp(-I); "clip": v = min(I, 1)


@dataclass
class Scene:
    seed: int = 1
    width: int = 1920
    height: int = 1080
    fps_num: int = 60000
    fps_den: int = 1001
    n_frames: int = 540
    hold: int = 2              # source animates at fps/hold
    hold_offset: int = 1       # first frame where a new image appears
    baseline_y: float = 539.0
    supersample: float = 2.0   # x samples per output pixel
    look: Look = field(default_factory=Look)
    layers: List[Layer] = field(default_factory=lambda: [Layer(), Layer(blur=2.0, gain=0.6)])
    strand_sets: List[StrandSet] = field(default_factory=lambda: [StrandSet()])
    strands: List[Strand] = field(default_factory=list)  # explicit extras
    jitter: StrandJitter = field(default_factory=StrandJitter)
    ambient: Ambient = field(default_factory=Ambient)
    packets: List[Packet] = field(default_factory=list)
    echo: Echo = field(default_factory=Echo)
    gain: GainGrid = field(default_factory=GainGrid)
    smudges: List[Smudge] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)  # scripted events, see src/events.py
    notes: str = ""

    # ---------------------------------------------------------------- helpers
    def frame_time(self, i: int) -> float:
        """Time of the image shown at output frame i (honours frame holding)."""
        if self.hold > 1:
            q = self.hold_offset + self.hold * ((i - self.hold_offset) // self.hold)
            i = max(q, min(self.hold_offset, self.n_frames - 1))
        return i * self.fps_den / self.fps_num

    def source_index(self, i: int) -> int:
        if self.hold > 1:
            q = self.hold_offset + self.hold * ((i - self.hold_offset) // self.hold)
            return max(q, min(self.hold_offset, self.n_frames - 1))
        return i


# --------------------------------------------------------------------------
# (de)serialisation
# --------------------------------------------------------------------------
def _build(cls, data: Any):
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data
    hints = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in data.items():
        if k not in hints:
            raise KeyError(f"unknown field {cls.__name__}.{k}")
        kwargs[k] = _convert(hints[k].type, v)
    return cls(**kwargs)


_TYPES = {c.__name__: c for c in (Packet, StrandJitter, Ambient, StrandSet, Strand, Layer,
                                  Glow, Echo, GainGrid, Smudge, Component, Look, Scene)}


def _convert(tp, v):
    tp = tp if isinstance(tp, str) else getattr(tp, "__name__", str(tp))
    tp = tp.replace("typing.", "")
    if tp in _TYPES:
        return _build(_TYPES[tp], v)
    if tp.startswith("List[") and tp[5:-1] in _TYPES and isinstance(v, list):
        return [_build(_TYPES[tp[5:-1]], e) for e in v]
    return v


def scene_from_dict(d: dict) -> Scene:
    return _build(Scene, d)


def scene_to_dict(s: Scene) -> dict:
    return dataclasses.asdict(s)
