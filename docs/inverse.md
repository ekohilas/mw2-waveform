# Inverse rendering: per-frame vector strands

Frame-accurate re-creation by fitting, for every one of the 270 content
frames, a set of vector strands (plus a soft haze layer) through a
differentiable copy of the frame model, directly on the scorer's metric.
The fitted parameters are derived data: they are regenerated from `ref/` by
committed code into `out/traces/` (gitignored) and never committed.

```
.venv/bin/pip install torch                                   # 2.8.0 runs on the venv's Python 3.9, MPS backend
.venv/bin/python -m tools.trace.metrics                       # validate the torch metrics against tools/score.py
.venv/bin/python -m tools.trace.fit --frames 21 101 345       # prototype frames (any output index -> its content frame)
.venv/bin/python -m tools.trace.fit --all --fit-look          # all 270 content frames -> out/traces/
out/scratch-style restart loop: --span a b --fit-look --skip-done   (see "Fit time")
.venv/bin/python -m tools.trace.fit --report                  # per-second summary of the saved fits
.venv/bin/python -m src.traces --score                        # 1920x1080 mp4 -> out/traces.mp4, then score.py
.venv/bin/python -m tools.trace.check_pipeline 101 345        # 960 render vs 1080p + x264 + area downscale
```

## Model (`src/traces.py`, shared by fitter and renderer)

All coordinates are in 960x540 "trace" pixels (the scoring resolution);
the renderer works at scale `s` (2 for 1920x1080).

* **Strand**: vertices `(x_j, y_j)` (one per column when traced, x free to
  move ±0.45 px), brightness `b_j ≥ 0`, width `u_j ∈ [0, 2]`. Between vertices
  the curve is a monotone cubic Hermite (PCHIP) `y(x)`: smooth at any
  resolution, no overshoot next to spikes. It is sampled by **arc length**
  (≤ 0.5 render px per sample, analytic `|dp/dt|`), each sample carrying mass
  `b·ds`, so near-vertical needles (270 px over 1–2 columns) are continuous
  strokes with the right brightness.
* **Splat**: bilinear into three core buffers by width (sharp, pre-blur
  `soft_sigma[0]`, pre-blur `soft_sigma[1]`; `u` interpolates linearly), so a
  strand can turn into the defocused "bloom blob" segments seen at crests.
* **Exposure**: `E_c = Σ_k A_kc · G(σ_k) * Σ_l G(soft_l) * core_l`, done in the
  Fourier domain with analytic Gaussian spectra (differentiable in σ, no
  kernel truncation, padded canvas so glow does not wrap). The kernel is
  peak-normalised: a horizontal line of `b = 1` gets `E_peak = Σ_k A_kc`.
* **Haze**: an 8-px-cell grid (68x120), bicubic-upsampled, tinted by
  `haze_rgb`, added to `E`: smudges, glow sheets, the low-frequency part of
  ghosting.
* **Tonemap**: `255·(1 − exp(−E) − offset)`, clipped.
* **Look** (`sigma`, `A`, `soft_sigma`, `offset`, `haze_rgb`) is fitted per
  frame (warm-started from the previous frame, projected to stay physical:
  A ≥ 0, σ ≥ 0.3). A single frozen look (median of the prototype fits) kept
  fine SSIM (0.969 vs 0.971 on c421) but lost 0.09 colour, because the
  colour/brightness relation drifts through the clip; per-frame looks fix it.
  The fitted looks put almost everything in the core component and the two
  width layers (σ ≈ 2 and ≈ 7) rather than in the fixed 3.3/7.6 glows.

Frame file `out/traces/cNNNN.npz`: concatenated `x, y, b, u` with per-strand
`lens`, the haze grid, and `info` JSON (look, fit metrics, timing).

## Loss (`tools/trace/metrics.py`)

A torch re-implementation of `tools/score.py`'s four metrics:

* luma SSIM exactly as skimage's defaults (7x7 uniform window via separable
  conv with scipy's symmetric "reflect" borders, sample covariance 49/48,
  K1 0.01, K2 0.03, data range 255), averaged over the lit mask (either
  frame's uint8 luma > 8, dilated 15x15; recomputed from the current render
  every 20 iterations). Validation against skimage/score.py on reference
  pairs: SSIM map max |Δ| ≤ 2.2e-4, frame means equal to 3 decimals
  (the remaining 4e-4 is the scorer's uint8 rounding of the luma);
* blurred SSIM, blurred colour L1 and column energy as in the scorer
  (Gaussian blurs with cv2's kernel sizes and reflect-101 borders).

`loss = (1−ssim) + 0.3(1−ssim_blur) + 0.3(1−color) + 0.1(1−energy)` + a
tiny L1 on the haze and on strand curvature. The render that enters the loss
is the **2x supersampled render, 2x2 area-averaged** (what the 1080p video
becomes after the scorer's `scale=960:540:flags=area`), so the 960 optimum
is the 1080p optimum.

## Fitting (`tools/trace/fit.py`, `tools/trace/init.py`)

Per content frame `c` (odd output index; `ref/frames/c.png`):

1. **Warm start**: the previous content frame's strands, haze and look are
   re-optimised for 60 iterations at scale 1; strands that no longer explain
   anything are pruned (b < 0.02), the rest capped at 8000 vertices by mass.
2. **Deconvolve** what is still unexplained: a non-negative per-pixel core
   image `L` (plus haze) with `tonemap(K*L + E_current) ≈ ref` (projected Adam,
   200 iterations, L1 prior). `L` is a sharp ~1-px line drawing of the frame.
3. **Trace** `L` by dynamic programming over columns (state = row, plus an
   "off" state so a path is a set of runs). A transition `(x, y) → (x+1, y')`
   collects the ridge-centre mass along the whole vertical span between the
   rows, from the map of its path direction (13 direction bins: mass ×
   alignment of the local Hessian line tangent with the step direction).
   That follows near-vertical needles while making zig-zags across parallel
   strands worthless. Each found path is erased (±1 px) and the next is
   traced until the gain drops below threshold; runs become strands with
   sub-pixel centroid y and `b` from the local core mass.
4. **Optimise** everything (y, x offsets, b, width, haze, look) with Adam on
   the loss: 150 iterations at scale 1, re-deconvolve + re-trace the residual,
   150 more, prune, then 150 at scale 2 (coarse-to-fine; dark vertices are
   frozen so they cannot wander, and dead strand ends/gaps are cut).

## Results

See "Prototype" and "Full run" below.
