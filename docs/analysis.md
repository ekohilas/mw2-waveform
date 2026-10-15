# Reference analysis: MW2 teaser 0:12–0:21 (oscilloscope section)

The spec the renderer is fitted against. All numbers were measured on
`ref/frames/0000–0539.png` (960x540, 8-bit sRGB, the scoring resolution) by the
scripts in `tools/analysis/` (run with `.venv/bin/python tools/analysis/<script>.py`).
**Units: px at 960x540.** The renderer works at 1920x1080, so double every
length (sigma, wavelength, amplitude, velocity) there. "Luma" is Rec.601
(`0.299R+0.587G+0.114B`), the same luma `score.py` uses. Frame `i` is source
time `12 + i*1001/60000` s.

| script | what it measures |
|---|---|
| `static_look.py` | baseline, cross-section fit, colour ramp, background, horizontal fade |
| `timeline.py` | per-frame energy/centroid/amplitude, holds/cuts, space-time envelope (`out/analysis/envelope.npz`) |
| `motion.py` | scrolling vs morphing, temporal correlation, travelling features |
| `smudges.py` | soft blobs / ghost streaks: detection, tracks, colour |
| `audio.py` | loudness envelope of the source audio vs visual amplitude |

## 1. Static look

### 1.1 Background
- Pure black: rows 0–99 and 440–539 average luma 0.035, only 0.1 % of
  pixels > 0.5 (all of them from smudges, see §4). No grain or noise floor; the
  source was limited-range video expanded to full range, so anything below
  ~Y 16 is crushed to 0. Render on `(0,0,0)`.
- Low values are smooth (no dither visible); the 8-bit histogram has the usual
  range-expansion gaps (values 7 and 248 never occur). Compression artifacts
  are minor and not worth modelling.

### 1.2 Baseline
- Centre **y = 269.5** (rows 269/270; the mean row profile peaks at 270, its
  centroid is 268.8 because upward excursions are slightly bigger). Flat across
  the frame: the per-column median peak row is 269–270 for x < 800, 271–273 at
  the far right.
- It is **a bundle of 2–3 strands**, not one line. In quiet columns (no wave
  activity) the vertical profile has 1 / 2 / 3 / 4+ resolved peaks in
  25 / 39 / 25 / 10 % of columns; strand separation 3–10 px (median 5); relative
  peak brightness 1 : 0.60 : 0.38. Strands wander ±2–3 px around y0 (peak row of
  quiet columns: 90 % within 266–273).
- Brightness along the baseline is beaded: per-column peak luma (σ=1 blur)
  p5/25/50/75/95 = 132/173/199/223/245 for x in 100–800; autocorrelation falls
  to 0.5 at ~10 px lag and 0.2 at ~20 px, i.e. bright beads/hotspots ~10–20 px
  long where strands cross. Hotspots saturate to near-white.

### 1.3 Cross-section (quiet columns, 10.6k samples)
Median vertical luma profile through the baseline (|dy| from the peak):

| \|dy\| px | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luma | 209 | 161 | 92 | 68 | 52 | 38 | 28 | 15 | 9 | 5 | 1.4 | 0 |

FWHM of the median profile is 3 px (per-sample p25/50/75 = 3/5/9 px; wider
when strands sit side by side). Fits (symmetrised):

- **3 Gaussians** (rms 0.45): `118.6·G(σ=0.99) + 70.3·G(σ=3.26) + 23.1·G(σ=7.65) − 1.9`, clipped at 0.
- Gaussian + exponential (rms 1.3): `77.7·G(σ=1.1) + 138·exp(−|dy|/4.1) − 3.0`.

So: a ~1 px sharp core, a ~3 px inner glow and a ~7.5 px outer glow; the glow
vanishes by |dy| ≈ 16–18 px. There is no wide (>20 px) bloom on the line
itself. Dim strands have relatively less glow (by peak brightness):

| peak luma | dy=0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 60–120 | 99 | 59 | 28 | 24 | 17 | 10 | 7 | 5 | 3 | 1 |
| 120–200 | 160 | 115 | 66 | 46 | 35 | 20 | 12 | 7 | 4 | 1 |
| 200–255 | 219 | 191 | 144 | 107 | 80 | 42 | 21 | 11 | 6 | 2 |

Thin off-baseline strands (wave packets) have vertical crossing FWHM
p25/50/75 = 2.3/3.7/6.5 px and peak luma p25/50/75 = 55/102/168 — many packet
strands are noticeably dimmer and thinner than the baseline bundle.

### 1.4 Colour
Colour is a function of brightness: deep green when dim, yellow-green in the
mid range, near-white in saturated cores. Median RGB per luma bin (all lit
pixels, every 10th frame) — usable directly as a 1-D LUT `luma -> RGB`:

| luma | R | G | B | R/G p10–p50–p90 |
|---|---|---|---|---|
| 2–5 | 0 | 5 | 0 | 0.00–0.00–0.33 |
| 10–15 | 4 | 19 | 0 | 0.00–0.19–0.53 |
| 20–30 | 12 | 35 | 1 | 0.05–0.33–0.61 |
| 40–60 | 28 | 67 | 10 | 0.22–0.43–0.64 |
| 80–100 | 59 | 117 | 28 | 0.35–0.51–0.68 |
| 120–140 | 93 | 165 | 49 | 0.44–0.56–0.68 |
| 160–180 | 127 | 210 | 74 | 0.50–0.60–0.70 |
| 200–220 | 178 | 246 | 108 | 0.63–0.72–0.81 |
| 220–235 | 212 | 251 | 143 | 0.77–0.85–0.93 |
| 245–256 | 249 | 250 | 240 | 0.96–1.00–1.02 |

G saturates first (~250 at luma 210), then R, then B — typical additive
"exposure" glow. **Core and halo have different tints** at equal luma (pixels
with luma 70–110): if the pixel is itself a dim strand core (7x7 max < 110)
R/G = 0.59, B/G = 0.38 (pale yellow-green); if it is the halo of a bright core
(7x7 max > 160) R/G = 0.46, B/G = 0.19 (saturated green).

**Recommended colour model** (fits the bright-baseline RGB cross-section to
rms 1.2/255 per channel): accumulate per-channel linear exposure, then
`out_c = 255·(1 − exp(−E_c)) − 2` (clip ≥ 0), with each strand contributing
`E_c(dy) = w · Σ_k A_kc · exp(−dy² / 2σ_k²)`:

| component | σ (px @960) | A_R | A_G | A_B | tint R/G, B/G |
|---|---|---|---|---|---|
| core | 1.0 | 0.93 | 2.46 | 0.53 | 0.38, 0.22 |
| inner glow | 3.3 | 0.43 | 0.92 | 0.21 | 0.46, 0.23 |
| outer glow | 7.6 | 0.035 | 0.143 | 0.00 | 0.24, 0.00 |

`w = 1` reproduces the bright baseline bundle (which is itself 2–3 strands, so
a single strand is roughly `w ≈ 0.5`). For single dim packet strands the fit
gives core `A ≈ (0.33, 0.53, 0.21)` — paler, R/G 0.61, B/G 0.40 — almost no
inner glow, outer glow `(0.05, 0.11, 0)`, offset 4.4. Because the tonemap
saturates G at ~250, overlapping strands go yellow-white automatically; never
write RGB directly.

### 1.5 Horizontal extent / fade
No brightness vignette for x < 800 (baseline peak luma ~220 flat from x = 2
to 800). Two edge effects:

| x | 0 | 2 | 10 | 80 | 120 | 240 | 480 | 640 | 720 | 800 | 820 | 830 | 840 | 880 | 950 | 959 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median column energy | 670 | 776 | 889 | 919 | 1638 | 2927 | 3972 | 3254 | 2814 | 1983 | 1393 | 981 | 946 | 597 | 404 | 310 |
| median baseline peak luma | 183 | 216 | 223 | 219 | 219 | 221 | 223 | 211 | 203 | 203 | 175 | 131 | 116 | 99 | 115 | 96 |

- **The bright baseline bundle ends at x ≈ 820–830** (soft over ~25 px);
  right of it only 1–2 dim strands remain (peak luma ~100–120), fading further
  toward x = 959. Left edge: the line runs to x = 0 at full brightness (1–2 px
  roll-off).
- **Wave activity is confined to x ≈ 100–820**, centred ≈ 460–480 with spread
  σ ≈ 170 px (§3); columns x < 90 and x > 830 rarely carry packets (median
  column energy there ≈ the baseline-only value, 600–1000).

## 2. Line structure (wave packets)

- **Strand count.** Quiet columns: the 2–3-strand baseline bundle (§1.2).
  Active columns (x 100–820 with off-baseline energy): median **2 off-baseline
  strand crossings per column** (0/1/2/3/4/5+ in 4/27/36/15/6/12 %), plus the
  baseline bundle — i.e. a packet is ~3–5 overlapping strands. Strands do not
  look like scaled copies of one waveform: each has its own shape/phase, they
  share only the envelope (where and how tall). Strands are drawn on top of an
  undisturbed baseline bundle (the baseline never disappears under a packet).
- **Brightness.** Off-baseline strand peak luma p10/25/50/75/90 =
  34/55/102/168/205 (baseline bundle: median 209). Roughly 1/3 of strands are
  dim (<60), 1/3 bright (>140, glow saturating to yellow-white where they
  cross). Near-vertical segments are ~0.7x as bright as horizontal/diagonal ones
  (ridge luma median 99 at 75–90° vs 120–158 at 0–60°): constant-width strokes,
  no oscilloscope-style beam-speed dimming beyond that.
- **Symmetry.** Statistically symmetric about y0: column max extents (luma>40)
  up/down p50 = 7.5/8.5, p90 = 33/29, p99 = 118/92 px; energy above/below ratio
  median 0.96 (p10–p90 0.62–1.54). Rare tallest excursions are slightly more
  often upward. Individual packets are not mirror-symmetric (independent
  strands go up and down).
- **Spatial frequency.** Crests of the upper envelope: spacing p10/25/50/75/90
  = 6/16/35/59/86 px; crest width at half prominence p25/50/75 = 5/13/21 px.
  Spectrum of the signed vertical centroid in active 256-px windows: power
  falls off from λ ≈ 128 px (envelope) through 85/64/51 px, down to 0.1 of peak
  at λ ≈ 32 px and 0.01 at λ ≈ 13 px. **Model wave components with
  λ ≈ 25–90 px (median ~40), plus a smooth envelope.**
- **Shapes.** Mostly smooth, rounded sinusoid-like wiggles (crest width/height
  ratio p25/50/75 = 0.17/0.44/0.84) with occasional sharp kinks/flat tops.
  Plus rare **needle spikes**: tall (90–270 px, can hit the frame edge) and very
  narrow (half-width ≤ 2–10 px, a single up-down stroke) — see §3.4.
- **Amplitude scale.** Per content frame, segments with smoothed envelope > 30 px:
  count p10/50/90 = 0/1/3, width p10/50/90 = 32/128/448 px, peak amplitude
  p10/50/90 = 32/56/144 px. Max excursion over the clip 270 px (frame edge).

## 3. Dynamics

### 3.1 Frame holds — content runs at 29.97 fps
Every frame `2k` (k ≥ 1) is an exact copy of frame `2k−1`, and frame 0 = frame 1
(270 held frames; mean |ΔL| < 0.05). New content appears only on odd frames.
**Render at 29.97 fps and show content index `c = max(1, 2·((i+1)//2) − 1)` on
output frame `i`.** This matters: between distinct content frames the
column-energy correlation is only 0.5 (below).

### 3.2 No cuts or flashes
Mean |ΔL| at the 269 content changes p1/50/99/max = 0.73/2.71/6.38/7.82 —
a continuum, no outliers. Frame mean luma ranges 2.69–9.71 and never jumps more
than 1.7 between content frames. No hard cuts, no flashes in the 540 frames.

### 3.3 No scrolling: the waveform redraws every content frame
- Global phase correlation (band y 210–330) between consecutive content frames:
  dx p10/50/90 = −0.46/+0.15/+1.41 px, low response — **no global scroll**.
- Local 220-px windows: best x-shift per content frame median −1 to −2 px
  (p25 −4…−7, p75 0…+2) with best correlation only ~0.55 — a weak
  **leftward drift (~30–60 px/s)** of the wave pattern, but mostly re-drawing.
- Temporal correlation (content-frame lags):

| lag (output frames) | 2 | 4 | 6 | 8 | 16 | 30 | 60 |
|---|---|---|---|---|---|---|---|
| column energy, high-passed | 0.50 | 0.18 | 0.09 | 0.04 | 0.02 | 0.03 | −0.01 |
| σ=6 blurred off-baseline image | 0.75 | 0.61 | 0.55 | 0.49 | 0.48 | 0.45 | 0.41 |

  Fine structure is new every ~2 content frames (66 ms); the blurred layout
  keeps a stable part (~0.45) from the fixed activity region (x 100–820).
  Packet tracks (envelope > 30 px, ≥ 50 % overlap frame to frame) live
  p50/75/90 = 2/4/9 content frames; envelopes of bursts persist 10–40 content
  frames (table below) while the strands inside them change every frame.
- **No phosphor trails / persistence**: dim pixels at t correlate equally with
  bright strands at t−k and t+k (0.11–0.13 for k = 1..4), i.e. no decaying copy
  of the previous frame. No motion blur along x (strokes stay 1 px-core sharp).

### 3.4 Travelling features (the only things that move coherently)
From the space-time amplitude map (`out/analysis/spacetime_ampfull.png`):

| feature | frames | path | speed | notes |
|---|---|---|---|---|
| needle pair | 339 → ~410 | both born at x ≈ 478 (f 339), split outward: x ≈ 478 ∓ 5.8·(f−339) | ±5.8 px/frame (±350 px/s) | full-height (270 px) at birth, decays to ~120–170 px by f 371–377, faint thin diagonals reach x = 0 / 960 by f ≈ 405–415 |
| left-travelling burst + smudge | ~357 → 475 | x ≈ 620 (f 360) → 520 (387) → 460 (405) → 383 (423) → 263 (441) → 117 (459) → 70 (471) | −5.1 px/frame mean, accelerating −3.5 → −7 | tall spikes (150–230 px) riding with a soft green smudge (§4) |
| needle cluster | 85–91, 131–135 | x ≈ 360–370, 405–411 | ~0 | short-lived narrow spikes 100–175 px |

### 3.5 Timeline (per 0.5 s; `timeline.py`)
`activity frac` = share of luma outside |dy| ≤ 8; `act x-centroid/spread` are
energy-weighted over that off-baseline luma; amplitudes are column max |dy| of
luma > 40.

| t (s) | frames | mean luma | activity frac | act x-centroid | act x-spread | p90 amp | max up | max dn |
|---|---|---|---|---|---|---|---|---|
| 12.0 | 0-29 | 2.74 | 0.17 | 509 | 210 | 15 | 36 | 34 |
| 12.5 | 30-59 | 2.99 | 0.17 | 478 | 214 | 15 | 56 | 36 |
| 13.0 | 60-89 | 4.32 | 0.33 | 455 | 169 | 35 | 198 | 156 |
| 13.5 | 90-119 | 6.03 | 0.47 | 472 | 171 | 68 | 202 | 198 |
| 14.0 | 120-149 | 6.45 | 0.49 | 475 | 172 | 61 | 170 | 132 |
| 14.5 | 150-179 | 5.77 | 0.44 | 468 | 179 | 47 | 158 | 108 |
| 15.0 | 180-209 | 4.64 | 0.34 | 467 | 188 | 31 | 116 | 74 |
| 15.5 | 210-239 | 4.53 | 0.36 | 473 | 179 | 53 | 120 | 184 |
| 16.0 | 240-269 | 4.96 | 0.35 | 453 | 180 | 34 | 76 | 134 |
| 16.5 | 270-299 | 4.45 | 0.37 | 452 | 172 | 41 | 110 | 80 |
| 17.0 | 300-329 | 4.57 | 0.31 | 438 | 174 | 41 | 156 | 126 |
| 17.5 | 330-359 | 7.17 | 0.50 | 482 | 158 | 88 | 270 | 250 |
| 18.0 | 360-389 | 8.03 | 0.55 | 501 | 170 | 74 | 270 | 180 |
| 18.5 | 390-419 | 7.69 | 0.52 | 490 | 161 | 74 | 266 | 150 |
| 19.0 | 420-449 | 6.45 | 0.48 | 418 | 144 | 57 | 180 | 176 |
| 19.5 | 450-479 | 5.16 | 0.38 | 396 | 196 | 60 | 222 | 196 |
| 20.0 | 480-509 | 4.34 | 0.28 | 468 | 173 | 29 | 114 | 86 |
| 20.5 | 510-539 | 4.10 | 0.24 | 437 | 188 | 26 | 132 | 88 |

Phases: **12.0–13.0 quiet** (baseline + small ripples ≤ 15–30 px);
**13.0–14.8 first loud burst**; **15.0–17.4 medium chatter**;
**17.5–19.0 loudest** (needle pair, smudges, full-height spikes);
**19.0–19.9 decaying, left-travelling burst**; **20.0–21.0 quiet tail**.

Major bursts (packet tracks ranked by peak amplitude x lifetime):

| frames | t (s) | x range (px) | peak amp (px) | content frames |
|---|---|---|---|---|
| 73-116 | 13.22-13.94 | 192-720 | 176 | 22 |
| 113-136 | 13.89-14.27 | 208-800 | 128 | 12 |
| 135-178 | 14.25-14.97 | 224-752 | 145 | 22 |
| 159-180 | 14.65-15.00 | 576-784 | 61 | 11 |
| 181-202 | 15.02-15.37 | 240-464 | 84 | 11 |
| 215-244 | 15.59-16.07 | 208-672 | 145 | 15 |
| 249-328 | 16.15-17.47 | 208-656 | 119 | 40 |
| 333-392 | 17.56-18.54 | 112-800 | 234 | 30 |
| 387-408 | 18.46-18.81 | 208-784 | 157 | 11 |
| 407-422 | 18.79-19.04 | 224-752 | 139 | 8 |
| 411-462 | 18.86-19.71 | 64-752 | 183 | 26 |
| 459-474 | 19.66-19.91 | 0-464 | 183 | 8 |
| 465-488 | 19.76-20.14 | 256-464 | 72 | 12 |
| 491-506 | 20.19-20.44 | 224-496 | 64 | 8 |
| 505-516 | 20.43-20.61 | 208-496 | 89 | 6 |

Full per-frame data: `out/analysis/timeline.csv`; per-frame, per-16-px-bin
up/down amplitude and off-baseline energy: `out/analysis/envelope.npz`
(`up`, `dn`, `act`, each 540 x 60) — **this is the fitting target for the
renderer's amplitude envelope** (fit e.g. a smooth A(x,t) per content frame).
