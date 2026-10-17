# MW2 teaser recreation (0:12–0:21)

Frame-accurate, higher-quality code re-creation of the visuals of the
[*Modern Warfare 2* teaser trailer](https://www.youtube.com/watch?v=iXEQTZEizaQ),
source time 0:12–0:21. The section is an abstract green oscilloscope-style waveform on black.

> Unofficial fan project, not affiliated with or endorsed by Activision or
> Infinity Ward. *Modern Warfare* is a trademark of Activision.

## Results

https://github.com/user-attachments/assets/31d172bf-0020-4b7c-8d95-98a52c842233

| render | command | fine SSIM | score |
|---|---|---|---|
| per-frame inverse rendering, 1080p H.264 | `python -m src.traces --score` | **0.982** | **0.985** |
| per-frame inverse rendering, 4K AV1 | `python -m src.traces --scale 4 --codec libsvtav1 --out out/traces_4k_av1.mp4 --score` | 0.981 | 0.984 |
| parametric style-only renderer (baseline) | `python -m src.render` | 0.429 | 0.666 |

Full history, one row per scored commit: `SCORES.md`.

## Rules
- The source video, extracted frames and audio never get committed
  (`*.mkv`, `*.wav`, `ref/`, `out/` are gitignored).
- Per-frame data derived from the local source (traced strands, extracted
  audio, fitted curves) is produced by committed scripts from the user's
  local copy into `out/`, rather than committed itself.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch 2.8 works on Python 3.9 (MPS)
# put the source .mkv in the repo root, then:
tools/extract_ref.sh            # -> ref/frames/0000..0539.png (960x540)
```

## Reproduce
```
.venv/bin/python -m tools.trace.fit --all --fit-look      # fit all 270 content frames -> out/traces/ (~3.7 h on MPS)
.venv/bin/python -m tools.trace.fit --report              # per-second summary of the fits
.venv/bin/python -m src.traces --score                    # 1080p -> out/traces.mp4 (~1 min), then score
.venv/bin/python -m src.traces --scale 4 --codec libsvtav1 --out out/traces_4k_av1.mp4   # 4K AV1 (~2.5 min)
.venv/bin/python -m src.traces --still 251                # one frame -> out/trace_still_0251.png
```
See `docs/inverse.md` for long runs (restarting the fitter every few frames).

## Scoring
```
.venv/bin/python tools/score.py out/traces.mp4
```
Compared at 960x540 (the source is 720p). Four per-frame metrics, all
normalised so an all-black render earns ~0 credit: lit-region SSIM ("fine
SSIM"), lit-region blurred SSIM, blurred colour difference, column-energy
profile. `score` is their mean. Target: fine SSIM 0.95.

Calibration baselines:
| render | score |
|---|---|
| all black | 0.09 |
| flat green line | 0.22 |
| the reference itself shifted by 1 s (right style, wrong timing) | 0.46 |
| the reference itself re-encoded with x264 | 0.975 |
| the same with encoder level compensation (`Mp4Writer(compensate=True)`) | 0.993 |

## Docs
| doc | contents |
|---|---|
| `docs/analysis.md` | measured spec of the look: line profile, colour model, timing (29.97 fps content, each image held 2 frames), events |
| `docs/renderer.md` | the parametric style-only renderer (`src/render.py`) and its scene format |
| `docs/audio.md` | why the waveform is **not** driven by the trailer's audio; the envelope prior it does give |
| `docs/inverse.md` | per-frame inverse rendering: model, loss, fitting, results, limits |

## Layout
| path | role |
|---|---|
| `src/traces.py` | trace model shared by fitter and renderer; renders `out/traces/` to video |
| `src/encode.py` | ffmpeg sinks: H.264 or 10-bit SVT-AV1, BT.709 tv-range, optional level compensation; PNG dirs |
| `src/render.py`, `waveform.py`, `frame.py`, `raster.py`, `events.py`, `params/` | parametric renderer |
| `src/fit/` | fitting tools for the parametric renderer |
| `tools/trace/` | inverse-rendering fitter, torch metrics (validated against the scorer), pipeline check |
| `tools/analysis/`, `tools/audio/` | measurement scripts behind `docs/analysis.md` and `docs/audio.md` |
| `tools/score.py`, `tools/extract_ref.sh` | scorer and reference extraction |

## Known issues
- Frame 251 (and possibly others) has a small magenta speck on the tallest
  peak: the per-frame colour fit is unconstrained and can push green below
  red/blue in very bright spots.
- The needle-pair frames (~17.7 s, worst content frame 345 at fine SSIM 0.95)
  are under-traced: strands are functions y(x), so near-vertical strokes a
  few pixels apart are hard to represent.
- The level compensation was measured for 8-bit x264; the 10-bit AV1 render
  loses ~0.005 colour against the 1080p H.264 one.
