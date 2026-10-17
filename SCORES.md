# Score history

| commit | description | ssim | ssim_blur | color | energy | score |
|---|---|---|---|---|---|---|
| 38c1852 | first full render: channels colour model, 3-strand bundle + 14 packet strands, 32-packet grid + gain grid coarse-fitted to per-bin extent/energy | 0.4145 | 0.6217 | 0.6500 | 0.9140 | 0.6501 |
| c8aa09d | baseline look tuned on quiet frames 0–60 (dimmer 3-strand bundle, stronger outer glow), activity-driven visibility option (off), refit grid; docs/renderer.md | 0.4176 | 0.6276 | 0.6451 | 0.9139 | 0.6511 |
| 57cdd46 | events module: needle pair born x 478 f 339, ±6.5 px/frame, ECG up/down strokes, ducks local packet strands; 17.5–20 s 0.5953 → 0.6026 | 0.4190 | 0.6329 | 0.6466 | 0.9138 | 0.6531 |
| bacf133 | smudges: 5 keyframed pulses (f 361–473, x 612→70, y 200→140) of the soft green glow on the left-travelling burst, with local ducking; 17.5–20 s 0.6158 | 0.4229 | 0.6394 | 0.6527 | 0.9120 | 0.6567 |
| db45f5d | burst: 5 plateau ('tombstone') bump pulses riding the smudges, ghost-streak fan f 455–471, smudges with sy2/pow/layer; 17.5–20 s 0.6201 | 0.4241 | 0.6419 | 0.6540 | 0.9117 | 0.6579 |
| 0e89d17 | per-key refit of smudge pulses and needle pair keys; 17.5–20 s 0.6220 | 0.4243 | 0.6424 | 0.6546 | 0.9124 | 0.6584 |
| 0b724b9 | tall spike event (x 521, f 379–393, ~270 px) + per-key x/a refit of the 5 burst pulses; 17.5–20 s 0.6250 | 0.4249 | 0.6438 | 0.6556 | 0.9129 | 0.6593 |
| 4c90b8d | look/structure: crisp 3-strand baseline, 8 packet strands (invisible when idle), 4-strand near-baseline rope, narrower core + pale-dim core gamma, black crush. **As committed (stale packets/gain): 0.5338.** Numbers here are after re-running `src.fit.coarse` (packets/gain not committed); old look under the same refit: 0.6520 | 0.4200 | 0.6512 | 0.6556 | 0.9065 | 0.6583 |
| 122fbda | merged look/structure + events; packets/gain refit (src.fit.coarse) against the merged look. Verified 1920x1080, 60000/1001, 540 frames | 0.4285 | 0.6607 | 0.6628 | 0.9114 | 0.6658 |
| 10baa49 | per-frame inverse rendering: vector strands + haze fitted to every content frame (tools/trace/fit.py -> out/traces/, per-frame look), rendered at 1920x1080 by `python -m src.traces` with encoder level compensation, crf 8. Verified 1920x1080, 60000/1001, 540 frames | 0.9818 | 0.9986 | 0.9656 | 0.9949 | 0.9852 |
| c185689 | 4K AV1: same traces rendered natively at 3840x2160 (`--scale 4 --codec libsvtav1`, 10-bit, crf 18, preset 4), 17 MB. Verified 3840x2160, 60000/1001, 540 frames. Colour slightly below 1080p (level compensation measured for x264) | 0.9813 | 0.9984 | 0.9614 | 0.9928 | 0.9835 |
