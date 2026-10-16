# Audio investigation: are the strands the audio waveform?

**Verdict: no.** The strand shapes are not a direct rendering of the trailer's
audio samples or spectrum. No window length, offset, direction, sign or
channel maps the audio onto the traced strands any better than audio from a
different moment does. The audio drives only the **loudness envelope**: how
much activity a frame has (r = 0.88), and it leads the picture by about
0–50 ms. It also cues the timing of the loudest event. Per-frame line shapes
have to come from inverse rendering. They cannot come from the audio.

Scripts are in `tools/audio/`. Every output goes to `out/audio/`, which is
gitignored.

| script | what |
|---|---|
| `extract.sh` | `out/audio/full.wav`: the whole track at 48 kHz, stereo, float32. Sample 0 is container time 0. |
| `trace.py` | DP strand tracer: 3 strands plus a signed centroid per content frame, written to `trace.npz` |
| `xcorr.py` | per-frame waveform-window search (FFT NCC over D, offset, channel, direction), with a null |
| `xcorr_fixed.py` | the same hypothesis with a single global mapping (signed NCC averaged over frames) |
| `spectrum_test.py` | spectrum hypothesis: x = frequency, height = band magnitude |
| `envelope.py` | the signal the audio does carry: envelope, lag, x-dependence, onsets. Writes `envelope_prior.npz`. |
| `events.py` | the audio around the needle pair and the left-travelling burst |

## 1. The track

- Opus, 48 kHz, stereo. **L and R differ**: over 12–21 s the sample correlation
  is 0.63, rms L/R = 0.0141/0.0131, and mid/side rms = 0.0123/0.0058. Every test
  below was run on L, R, mid and side.
- Over 12–21 s the energy splits by band as <60 Hz 4 %, 60–150 Hz 8 %,
  150–300 Hz 13 %, 300–1k 45 %, 1–3k 21 %, >3k 9 %. The spectral centroid
  (50–4000 Hz, 100 ms windows) has p10/50/90 = 481/790/1061 Hz. It is a
  voice-dominated mix.
- Timing: video pts are `n·1001/60000`. `ref/frames/0000` matches the decoded
  frame at pts 12.012–12.029, so ref frame *i* is about 12 s + 12–29 ms +
  *i*·1001/60000. The lags below are measured against `12 + i·1001/60000`; to
  convert them to true container time, subtract another 12–29 ms. Every search
  was wide enough to absorb this offset.

## 2. Tracing

`trace.py` runs on the 270 content frames (odd output frames). It treats luma
minus 40 as the reward, with the baseline band |dy| < 4 scaled by 0.2, and finds
the max-sum path y(x) under an L1 slope penalty of 2/px. The path uses an exact
vectorised L1 distance transform per column, about 3 s for all frames. The path
±5 px is then erased and the search repeats, giving 3 paths per frame. The paths
follow real strand segments, but they hop between strands where strands cross.
The packets are 3–5 overlapping strands, each with its own shape (analysis §2),
so no single clean strand exists. Each frame also stores a signed luma centroid
of the off-baseline light. Column range used: x 150–790 (the active region).

## 3. Waveform-window hypothesis (After Effects "Audio Waveform")

The model: `d(x) = A · a_lp(t_i + off + (x − 480)·k)` with `k = D/740` s/px.
That is a window of length D spread over the wave region, shifted by `off`. A
negative k means time runs right to left, and a negative A is a sign flip. The
audio is low-passed to keep features of 6 px or more, which plays the role of
the effect's decimation to "Displayed Samples". The search covers 48 log-spaced
D from 5 ms to 3 s, offsets within ±400 ms (centred window, window ending at t,
and anything in between), channels L/R/M/S, both directions, both signs, and all
3 traces.

AE's Audio Waveform has these controls: Audio Layer, Start/End Point, Path,
Displayed Samples, Maximum Height, Audio Duration (ms), Audio Offset (ms),
Thickness, Softness, Random Seed (Analog), Inner/Outer Color, Waveform Options
(mono/L/R) and Display Options (Digital / Analog lines / Analog dots). These
are from memory; Adobe's help page returned 403. Every one of these mappings is
a special case of the model above, with possibly several jittered copies, and
any one of those copies would match one of the traced strands.

### 3a. Best fit per frame (`xcorr.py`)

| search | best \|r\| per frame p10 / p50 / p90 | frames above the null's p95 |
|---|---|---|
| real audio at t | 0.602 / 0.665 / 0.755 | — |
| null: same search, audio at t − 3 s (same speech passage) | 0.596 / 0.655 / 0.743 | **4.4 %** (5 % is chance) |
| null: audio at t + 40 s (different material) | 0.531 / 0.597 / 0.687 | 28.5 % |

Mean best \|r\| per window length, real vs the −3 s null:

| D (ms) | 7.5 | 13 | 22 | 38 | 66 | 114 | 197 | 340 | 586 | 1010 | 1740 | 3000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 0.63 | 0.61 | 0.57 | 0.52 | 0.46 | 0.42 | 0.36 | 0.32 | 0.25 | 0.20 | 0.21 | 0.28 |
| null | 0.60 | 0.59 | 0.58 | 0.55 | 0.50 | 0.45 | 0.38 | 0.32 | 0.24 | 0.19 | 0.23 | 0.27 |

The best per-frame parameters are inconsistent across frames. D piles up at the
5 ms lower edge of the grid, where a few samples of high-frequency audio can
bend into any smooth curve. The offsets are uniform over ±400 ms. The best
per-frame match (r ≈ 0.6–0.85) is therefore exactly the chance level of a
search this wide. The +40 s null scores lower only because that part of the
trailer has a different spectrum.

### 3b. One mapping for all frames (`xcorr_fixed.py`)

A real AE mapping uses the same D, offset, sign and direction in every frame.
For each D, `R(D, off) = mean_i r_i(D, off)` is computed over all 270 frames and
maximised over offset, channel and direction:

| D (ms) | 5 | 10 | 20 | 39 | 66 | 100 | 197 | 390 | 1010 | 3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| \|R\| trace, real | 0.042 | 0.037 | 0.038 | 0.028 | 0.023 | 0.023 | 0.017 | 0.015 | 0.010 | 0.030 |
| \|R\| trace, null | 0.050 | 0.039 | 0.044 | 0.052 | 0.065 | 0.034 | 0.016 | 0.009 | 0.014 | 0.028 |
| \|R\| centroid, real | 0.044 | 0.043 | 0.040 | 0.030 | 0.029 | 0.025 | 0.018 | 0.013 | 0.008 | 0.040 |
| \|R\| centroid, null | 0.053 | 0.057 | 0.065 | 0.055 | 0.073 | 0.035 | 0.016 | 0.007 | 0.010 | 0.028 |

All values are noise, below 0.06 and no higher than the null, and no D has a
peak. A real mapping would give |R| ≈ 0.3–0.9 even with a strand-hopping trace.

### 3c. Independent spectral evidence

The dominant spatial wavelength of the strands barely changes: the spectral
centroid of the traces gives λ p10/50/90 = 40/47/56 px. Over the same frames
the audio's spectral centroid ranges over 481–1061 Hz (×2.2), and the two have
Pearson r = −0.03 (−0.03 on loud frames only). If the strands were the waveform
under any fixed D, their wavelength would scale with the audio's frequency
content. They behave like band-limited noise with a fixed λ ≈ 45 px under an
envelope.

## 4. Spectrum hypothesis (`spectrum_test.py`)

The model: x = frequency (linear or log, f_lo 20–200 Hz, f_hi 1–16 kHz) and
column extent = |STFT| (N = 1024–8192, linear or dB, lag −100…+50 ms). Both
sides were taken as deviations from the clip-mean profile. The best mean r
across frames is **−0.099** for real audio and 0.084 for the null. Rejected. It
was already unlikely: a spectrum is non-negative, but the strands swing up and
down independently.

## 5. Other variants

- **Stereo, mid/side, sign flips, mirrored x, centred vs trailing window**: all
  inside the searches above. None scores above the null.
- **Per-strand filtering and time-shifted copies**: each traced strand was
  tested on its own against band-limited audio at every offset and D. Any
  filtered or shifted copy is a case of that search, apart from the filter
  shape. Nothing found.
- **Scrolling envelope (Trapcode-Form-style delay propagation)**: the activity
  map does not travel. Correlating act(x, t) with act(x + v·k, t + k) peaks at
  v ≈ 0 on both halves of the frame; the left half leans slightly leftward and
  the right half slightly rightward (k = 2–8 output frames). The best per-x-bin
  audio lag (below) has no linear trend in x.
- **Envelope-modulated noise**: this is what fits (§6). It is what the current
  renderer already does.

## 6. What the audio does give (`envelope.py`)

**Global loudness envelope.** Best Pearson r between visual activity per
content frame and audio log-RMS, maximised over window (10–200 ms) and lag
(±200 ms, negative = audio earlier):

| band | log act-energy | mean amplitude | p90 amplitude |
|---|---|---|---|
| full | **0.876** (200 ms, −10 ms) | **0.795** (33 ms, −40 ms) | **0.714** (33 ms, −40 ms) |
| <300 Hz | 0.646 (200 ms, +30) | 0.630 | 0.506 |
| 300–3k | 0.849 (200 ms, −50) | 0.712 | 0.646 |
| 1k–4k | 0.791 (200 ms, −120) | 0.655 | 0.594 |
| >4k | 0.772 (10 ms, −60) | 0.665 | 0.615 |

The full band gives the best match. The lag curve is broad: r ≥ 0.85 from −100
to +40 ms, and amplitude peaks at −40 ms (audio about 2–3 output frames ahead),
in line with the analyst's 3 frames. Prior fits (full band, 200 ms window,
−10 ms lag, `env` = ½·ln(mean square)): `log1p(Σ act) ≈ 1.382·env + 19.98` and
`p90 amp ≈ 59.67·env + 337.6` px at 960 width. Per-frame predictions are in
`out/audio/envelope_prior.npz`.

**No spatial information.** Per 16-px bin, the correlation of log act with the
same envelope is r = 0.6–0.8 across x 150–780 (0.4 at the edges). The best lag
per bin falls between −170 and +140 ms with no trend. The audio sets *how much*
activity a frame has, not *where* it is. Where it falls comes from the fixed
spatial hump (x ≈ 470, σ ≈ 170) and from the visual data.

**Onsets.** There are 21 audio onsets (rises of the log envelope) and 23 visual
onsets. 48 / 67 / 86 % of audio onsets have a visual onset within ±1 / 2 / 3
content frames, against 23 / 35 / 46 % for random placement. Packet births
therefore follow syllable onsets, give or take a frame or two.

## 7. The moving events (`events.py`)

- **Needle pair (born frame 339, x 478, ±5.8 px/frame).** This is audio-cued:
  the strongest onset of the section falls at frames 331–333 (17.52–17.56 s,
  the envelope jumps from −46 to −34 dB), and the loudest moment of the clip is
  frames 341–343 (−27 dB). The motion is not audio-driven, though. A transient
  sliding through a waveform window moves one way at 740/D px/s, while the
  needles split symmetrically in both directions. The 348 px/s that would imply
  (D ≈ 2.1 s) would also scroll the whole waveform, and there is no global
  scroll (§5). This is a separate keyframed layer triggered on the hit.
- **Left-travelling burst (frames 357–475).** Its speed of −307 px/s would need
  D ≈ 2.4 s, again with no matching global scroll. The audio during the burst
  has no special transient: the envelope stays between −33 and −42 dB, and the
  only flux peaks are at frames 377, 405 and 437. A separate animated layer,
  timed to the loud phrase.

## 8. Recommendations

- Don't build `src/audio_wave.py`. There is no mapping to drive strand
  geometry. Fine SSIM 0.95 needs per-frame inverse rendering of strand shapes.
- To the inverse-rendering agent: use the audio as a prior on **per-frame
  activity**, not shape. Use `envelope_prior.npz` (r 0.88 for log energy, 0.80
  for mean amplitude, audio about 40 ms ahead), the onset timings (packets
  appear within about 1–2 content frames of audio onsets), and the needle-pair
  trigger at the 17.52–17.72 s hit. The strands themselves behave like
  band-limited noise with λ ≈ 40–56 px (median 47 px at 960 width) under that
  envelope.
