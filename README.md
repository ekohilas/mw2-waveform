# MW2 teaser recreation (0:12–0:21)

Code-only recreation of the visuals of the *Modern Warfare 2* teaser
(https://www.youtube.com/watch?v=iXEQTZEizaQ), 0:12–0:21, rendered at
1920x1080, 59.94 fps (540 frames). The section is an abstract green
oscilloscope-style waveform on black.

## Rules
- The source video, extracted frames and audio never get committed
  (`*.mkv`, `ref/`, `out/` are gitignored).
- Goal (updated): a frame-accurate, higher-quality re-creation. Per-frame data
  derived from the local source (traced strands, extracted audio, fitted
  curves) is allowed, but it is produced by committed scripts from the user's
  local copy into `out/` (gitignored) rather than committed itself.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the source .mkv in the repo root, then:
tools/extract_ref.sh            # -> ref/frames/0000..0539.png (960x540)
```

## Scoring
```
.venv/bin/python tools/score.py out/render.mp4
```
Compared at 960x540. Four per-frame metrics, all normalised so an all-black
render earns ~0 credit: lit-region SSIM, lit-region blurred SSIM, blurred
colour difference, column-energy profile. `score` is their mean. Target: 0.95.

Calibration baselines:
| render | score |
|---|---|
| all black | 0.09 |
| flat green line | 0.22 |
| the reference itself shifted by 1 s (right style, wrong timing) | 0.46 |
| the reference itself re-encoded (ceiling) | 0.98 |

Score history is tracked in `SCORES.md`.

## Status
Current best: **0.985** overall, **fine SSIM 0.982** (see `SCORES.md`), via
per-frame inverse rendering (`docs/inverse.md`):
```
.venv/bin/python -m tools.trace.fit --all --fit-look   # ~3.7 h on MPS -> out/traces/
.venv/bin/python -m src.traces --score                 # 1080p -> out/traces.mp4 (~1 min)
```
The waveform is not audio-driven (`docs/audio.md`). The earlier parametric
renderer (`src.render`, 0.666) remains as a style-only baseline.
