# Score history

| commit | description | ssim | ssim_blur | color | energy | score |
|---|---|---|---|---|---|---|
| 38c1852 | first full render: channels colour model, 3-strand bundle + 14 packet strands, 32-packet grid + gain grid coarse-fitted to per-bin extent/energy | 0.4145 | 0.6217 | 0.6500 | 0.9140 | 0.6501 |
| c8aa09d | baseline look tuned on quiet frames 0–60 (dimmer 3-strand bundle, stronger outer glow), activity-driven visibility option (off), refit grid; docs/renderer.md | 0.4176 | 0.6276 | 0.6451 | 0.9139 | 0.6511 |
| 57cdd46 | events module: needle pair born x 478 f 339, ±6.5 px/frame, ECG up/down strokes, ducks local packet strands; 17.5–20 s 0.5953 → 0.6026 | 0.4190 | 0.6329 | 0.6466 | 0.9138 | 0.6531 |
| bacf133 | smudges: 5 keyframed pulses (f 361–473, x 612→70, y 200→140) of the soft green glow on the left-travelling burst, with local ducking; 17.5–20 s 0.6158 | 0.4229 | 0.6394 | 0.6527 | 0.9120 | 0.6567 |
