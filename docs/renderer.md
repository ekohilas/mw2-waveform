# Renderer

Generative recreation of the 0:12–0:21 oscilloscope section. Pure code +
a compact parameter file; the renderer never reads `ref/`, the source video or
audio.

## Usage
```
.venv/bin/python -m src.render                    # 1920x1080 -> out/render.mp4
.venv/bin/python -m src.render --preview --score  # 960x540 PNGs -> out/preview/, then score
.venv/bin/python -m src.render --still 360 [--preview]   # one frame -> out/still_0360.png
.venv/bin/python -m src.render --scene other.json --out out/x.mp4
.venv/bin/python tools/score.py out/render.mp4
```
Speed (10 cores): full render incl. x264 ≈ 7–15 s; preview ≈ 5 s.
Single-threaded ≈ 55 ms/frame at 1080p, 15 ms at 960x540. Only the 270
unique content images are rendered (the source runs at 29.97 fps; output frame
`i` shows content `max(1, 2*((i+1)//2) - 1)`, see `Scene.hold*`).

## Architecture (`src/`)
| module | role |
|---|---|
| `params/schema.py` | dataclass scene description, JSON (de)serialisation, keyframe evaluation |
| `params/scene.json` | the current parameter set (default scene) |
| `waveform.py` | `WaveModel`: expands strand families, evaluates every strand's `y(x)` and brightness `b(x)` at time t |
| `raster.py` | anti-aliased polyline rasteriser (difference-array vertical spans, exact fractional coverage, O(segments + pixels)); Gaussian blur (downsampled for big σ); smudge blobs |
| `frame.py` | `FrameRenderer`: layers -> core buffer -> per-channel exposure (3-Gaussian kernel with RGB tints) + smudges -> `255·(1−exp(−E)) − offset` |
| `color.py` | legacy single-channel tonemap + colour ramp (`look.mode = "ramp"`) |
| `encode.py` | ffmpeg pipe: libx264 crf 16, yuv420p, BT.709 tv-range, 60000/1001; PNG dir writer |
| `render.py` | CLI; spawn-pool over cores, ordered writer that duplicates held frames |
| `fit/` | fit-time tools (read `ref/`; never imported by the renderer) |

Per frame: for each strand, `y = baseline_y + strand.y − displacement`, where
displacement = ambient wiggle (per-strand random sinusoids) + Σ packets.
Strands are rasterised as supersampled polylines (mass ∝ brightness ×
thickness × arc length, so steep spikes keep their brightness) into one buffer
per *layer* (sharp / soft, each with its own pre-blur). The summed core buffer
`L` is convolved with each `look.components[k]` Gaussian (peak-normalised, so
a line with `b·th = 1` gets `E_peak = rgb_k`), giving per-channel exposure
`E`; smudges add `smudge_rgb · blob` to `E`; output is `1 − exp(−E) − offset`.

## Parameter format (`src/params/scene.json`)
All lengths are **design px at 1920x1080**, times are **seconds from the
section start** (frame i at `i·1001/60000`). Anything animated is keyframed:
a `t` array plus values that are either scalars (constant) or arrays aligned
with `t`, linearly interpolated.

- `baseline_y`, `seed`, `supersample`, `hold`, `hold_offset`
- `look`: `mode` ("channels"), `components: [{sigma, rgb:[R,G,B]}]`,
  `offset`, `exposure`, `smudge_rgb`
- `layers: [{blur, gain}]` — layer 0 sharp, 1 soft
- `strand_sets: [StrandSet]` — procedural families expanded deterministically
  from `(seed, set.seed)`: `count, y_spread, brightness(_spread),
  thickness(_spread), soft_fraction, knot_amp/lam/speed` (beading along x),
  `response` (packet gain), `carrier` (probability a strand carries a given
  packet at full gain, else `jitter.idle_gain`), `x_end, x_fade,
  end_fraction` (strands ending at the bundle end), `vis_amp, vis_floor`
  (brightness follows the packet amplitude the strand carries — packet strands
  disappear when idle). `strands: [Strand]` adds explicit strands.
- `jitter`: per-(strand, packet) lognormal gain, phase, wavelength and x
  jitter; seeded by packet *index*, so appending packets is stable.
- `ambient`: idle wiggle (`amp`, λ range, speed, `n`, keyed `gain`).
- `packets: [Packet]` — the main fitting surface. One packet:
  `t[]`, keyed `x, w (σ), a (peak px, up), lam`, plus `phase, speed (cycles/s),
  sharp (|sin|^p peaks), harm2, env_pow, jitter`. Amplitude is 0 outside
  `[t0, tN]`. Displacement on strand s:
  `g_s·a·exp(−½|u|^env_pow)·shape(sin(2π(x−x_c)/(λ·l_s) + phase + 2π·speed·t + δ_s))`.
- `gain: {t[], x[], g[t][x]}` — bilinear brightness multiplier over (t, x).
- `smudges: [{t[], x, y, sx, sy, a, angle}]` — anisotropic Gaussian blobs in
  exposure units, zero outside their key range.
- `echo` — optional ghost copies at earlier times (off by default).

`save_scene()` writes compact JSON (one packet per line); keep the file well
under 1 MB.

## Fitting
`src/fit/common.py` has fit-time helpers: coarse per-bin measurements,
parallel render+measure, and `score_subset(scene, frames)` which runs the exact
`tools/score.py` metrics on a frame subset in parallel (~2 s for 54 frames).

`.venv/bin/python -m src.fit.coarse` (first pass): keeps the scene's look,
replaces `packets` with a grid of 32 fixed packets (one per 60-px column,
amplitude keyed at ~15 Hz) and `gain` with a matching (t, x) grid, then
iterates render → measure → update so the per-bin rms vertical extent and
summed luma match the reference. A finer fitter can replace the grid with
free packets (any number, own keyframes) without changing the renderer.

## Scripted events (`src/events.py`)
The coherent moving features of docs/analysis.md §3.4/§4 are compact keyframed
objects in `scene.events` (one dict each, `type` + `t[]` keys, ≤ ~10 keys) and
`scene.smudges`; nothing is traced from the reference. Hooks: `FrameRenderer`
builds an `EventModel`; `strand_layers` multiplies waveform brightness by
`events.duck(...)` and calls `events.raster(...)`; `smudge_buffer` delegates to
`events.smudge_buffer`.

| type | what | keys |
|---|---|---|
| `needle_pair` | two ECG needles born at `x0`, centres `x0 ± d(t)`: up-stroke to `up`, down-stroke to `dn` `gap` px further out | `d, up, dn, b`; `x0, w, gap, th, layer` |
| `bump` | `n` flat-topped spikes `a·exp(-½|dx/w|^p)` travelling along `x(t)`, offsets re-drawn every `redraw` s | `x, a, b`; `n, w, spread, jit, p, down, seed` |
| `streaks` | faint fan of `n` lines from `x(t)` at heights `y·f` down to the baseline `len` px to the right | `x, y, len, b`; `n, fmin, wig, lam` |

Any event (and any smudge) may set `duck` (0..1): inside its footprint the
*displaced* parts of the waveform strands are dimmed, so the event replaces the
packet energy the coarse grid had put there instead of adding to it.
Smudges gained `sy2` (sigma below the centre), `pow` (flat-top exponent) and
`layer` (−1: exposure-space blob tinted `look.smudge_rgb`; ≥ 0: light added to
that layer's core buffer, so it gets the lines' colour/glow pipeline).

Current set (17.5–20 s): needle pair f 339–412; burst = five plateau pulses
(`burst1..5`) + a tall spike (f 379–393) travelling x 612 → 70 (960-px), each
pulse paired with a smudge; ghost-streak fan f 455–471. Fit with
`src/fit/events.py` (`Scorer` on content frames 331–479, `coord_search`).
After changing events, re-running `src.fit.coarse` lets the packet grid
account for the event energy (tested with default args: 0.659 → 0.665 total,
17.5–20 s 0.625 → 0.634; the same refit without events gives 0.655 / 0.597;
keep ducking on, it still helps after a refit).
