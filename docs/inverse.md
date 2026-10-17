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
# long runs: restart the process every few frames (see "Fit time"), e.g.
#   for a in $(seq 1 12 539); do .venv/bin/python -m tools.trace.fit --span $a $((a+10)) --fit-look --skip-done; done
.venv/bin/python -m tools.trace.fit --report                  # per-second summary of the saved fits
.venv/bin/python -m src.traces --score                        # 1920x1080 mp4 -> out/traces.mp4, then score.py
.venv/bin/python -m src.traces --scale 4 --codec libsvtav1 --out out/traces_4k_av1.mp4 --score   # 3840x2160 10-bit AV1
.venv/bin/python -m src.traces --still 251                    # one output frame -> out/trace_still_0251.png
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

Fine SSIM is the scorer's `ssim`. "fit" = exact `score.frame_metrics` of the
fitter's own 2x render (uint8, no codec); "mp4" = the frame in the full
`out/traces.mp4` through `tools/score.py` (1080p render, x264, decode, area
downscale).

### Prototype (8 frames, cold start, look fitted per frame)

| content frame | what | strands | fine SSIM |
|---|---|---|---|
| 21 | quiet | 7 | 0.9525 |
| 101 | busy, first burst | 32 | 0.9594 |
| 201 | medium chatter | 18 | 0.9751 |
| 251 | busy | 22 | 0.9712 |
| 345 | needle pair | 65 | 0.9747 |
| 381 | burst + smudge | 64 | 0.9721 |
| 421 | burst + smudge | 117 | 0.9714 |
| 501 | decaying tail | 24 | 0.9770 |

Before per-frame looks, the fixed analysis look gave 0.82 on c101
(glow profile and colour wrong everywhere along the lines).

### Full run (all 270 content frames, warm-started chain)

| content frame | strands | fit SSIM | mp4 SSIM | mp4 frame score |
|---|---|---|---|---|
| 21 | 33 | 0.9884 | 0.9866 | 0.9881 |
| 101 | 114 | 0.9806 | 0.9794 | 0.9839 |
| 201 | 44 | 0.9896 | 0.9881 | 0.9900 |
| 251 | 72 | 0.9896 | 0.9879 | 0.9878 |
| 345 | 187 | 0.9498 | 0.9487 | 0.9626 |
| 381 | 142 | 0.9766 | 0.9754 | 0.9808 |
| 421 | 226 | 0.9695 | 0.9681 | 0.9776 |
| 501 | 68 | 0.9904 | 0.9889 | 0.9885 |

Warm starting is worth a lot: accumulated faint strands carried over from
earlier frames (ghost sheets persist for several content frames, analysis
section 4) push quiet frames from ~0.95–0.97 cold to ~0.988.

Per second (fit metrics, 2x render, no codec):

| t | fine SSIM | ssim_blur | color | energy | strands | fit s/frame |
|---|---|---|---|---|---|---|
| 12–13 | 0.9879 | 0.9993 | 0.9751 | 0.9986 | 29 | 39 |
| 13–14 | 0.9820 | 0.9987 | 0.9659 | 0.9969 | 85 | 46 |
| 14–15 | 0.9848 | 0.9989 | 0.9699 | 0.9973 | 101 | 50 |
| 15–16 | 0.9872 | 0.9992 | 0.9721 | 0.9980 | 75 | 45 |
| 16–17 | 0.9885 | 0.9990 | 0.9715 | 0.9982 | 86 | 49 |
| 17–18 | 0.9803 | 0.9976 | 0.9581 | 0.9962 | 112 | 54 |
| 18–19 | 0.9736 | 0.9979 | 0.9524 | 0.9972 | 174 | 58 |
| 19–20 | 0.9796 | 0.9987 | 0.9634 | 0.9982 | 188 | 57 |
| 20–21 | 0.9902 | 0.9993 | 0.9724 | 0.9987 | 64 | 42 |
| all | **0.9838** | 0.9988 | 0.9668 | 0.9977 | | 3.7 h total |

Full clip through `tools/score.py` (`python -m src.traces --score`):
**ssim 0.9818, ssim_blur 0.9986, color 0.9656, energy 0.9949, score 0.9852**
(previous best 0.4285 / 0.6658). Worst frame c345 (0.949). Render of all
540 frames at 1920x1080 + x264: 76 s (MPS).

4K: the strands are vector curves, so `--scale 4` draws them natively at
3840x2160 (not upscaled). With `--codec libsvtav1` (10-bit yuv420p10le,
crf 18, preset 4) the 17 MB file scores **ssim 0.9813, ssim_blur 0.9984,
color 0.9614, energy 0.9928, score 0.9835** after the scorer's downscale;
render + encode ≈ 2.3 min. The small colour loss vs 1080p is because the
level compensation was measured for 8-bit x264, not 10-bit AV1.

### 960 optimum vs 1080p delivery

The fit renders at 2x and area-averages, so the remaining gap is quantisation
and the codec. `tools/trace/check_pipeline.py` on c419/c421 (fit SSIM 0.9653 /
0.9751):

| path | c419 | c421 |
|---|---|---|
| 1080p uint8, area downscale (no codec) | 0.9653 | 0.9751 |
| x264 crf 16, as src.encode did | 0.9429 | 0.9556 |
| x264 crf 16 + level compensation | 0.9624 | 0.9705 |
| x264 crf 8 + level compensation (used) | 0.9636 | 0.9729 |

The big one: the encode (rgb→yuv tv range) + the scorer's decode shift
every channel by about −3.7 / −2.0 / −3.7 (B, G, R), measured with flat
random-colour patches. `Mp4Writer(compensate=True)` pre-adds that. On the
re-encoded reference it lifts the ceiling from 0.975 to 0.993 (fine SSIM
0.978 → 0.992), so **the statistical renderer should turn it on too**. Over
the whole clip the fit → mp4 loss is 0.0020 fine SSIM.

### What limits it

* **Overlapping near-vertical strokes** (needle pair, c339–360; tall burst
  spikes): several 1–2 px apart strokes in the same columns. A strand is a
  function y(x) with one vertex per column, and the DP tracer takes one row
  per column per path, so parallel steep strokes and the blobs on needle
  flanks are under-traced. c345 is the worst frame (0.95). Fix: allow x to
  move freely (curves parameterised by arc length) or trace steep strokes
  column-transposed.
* **Faint ghost sheets / stippled strokes** (luma 5–30, comb and dotted
  textures, strongest 18–20 s): inside the lit mask they cost SSIM, too fine
  for the 8-px haze grid and too faint/fragmented for the tracer. Warm-started
  strands capture part of them.
* **Source compression noise**: in dark areas near the luma-8 threshold the
  reference shows 8x8 macroblock steps that no strand model reproduces
  (visible as square spots in SSIM maps of quiet frames).
* **Colour** (0.967) is the weakest metric: the colour model is three
  RGB-tinted Gaussians + per-frame tints; dim cores are paler than bright
  ones (analysis 1.4) and the model has no brightness-dependent tint.
* **Colour artefacts**: the unconstrained per-frame tints can push green
  below red/blue in very bright spots, e.g. a ~130 px magenta speck on the
  tallest peak of frame 251 (1080p x≈828–850, y≈471–479). Too small to move
  the score; fix by constraining tints (G ≥ R, B) and refitting affected frames.
* **Looks drift**: per-frame looks are unconstrained between frames; late
  frames moved the second width layer to a very wide blur (σ up to ~180 px),
  effectively a second haze. Harmless for the score, but the parameters are
  less physical than the strands.

### Fit time

~45 s per content frame on MPS (M-series GPU, 1 process), 3.7 h for all 270.
Tracing (deconvolution + DP) is ~5 s; the rest is ~660 Adam iterations.
Two traps: a long-lived process slowed down steadily on MPS (new tensor
shapes every iteration from variable sample counts and boolean indexing), now
fixed with bucketed sample buffers and `torch.where`. Remaining slow drift is
side-stepped by restarting the process every 6 content frames
(`--span … --skip-done` warm-starts from the saved previous frame).
