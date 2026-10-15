# MW2 teaser recreation (0:12–0:21)

Code-only recreation of the visuals of the *Modern Warfare 2* teaser
(https://www.youtube.com/watch?v=iXEQTZEizaQ), 0:12–0:21, rendered at
1920x1080, 59.94 fps (540 frames). The section is an abstract green
oscilloscope-style waveform on black.

## Rules
- The source video, extracted frames and audio never get committed
  (`*.mkv`, `ref/`, `out/` are gitignored).
- The renderer is generative code. Fitted parameters (keyframes, envelopes,
  wave components) are fine; embedding reference pixels or audio samples is not.

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
