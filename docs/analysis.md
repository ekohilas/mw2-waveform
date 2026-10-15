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
