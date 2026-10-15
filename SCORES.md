# Score history

| commit | description | ssim | ssim_blur | color | energy | score |
|---|---|---|---|---|---|---|
| 38c1852 | first full render: channels colour model, 3-strand bundle + 14 packet strands, 32-packet grid + gain grid coarse-fitted to per-bin extent/energy | 0.4145 | 0.6217 | 0.6500 | 0.9140 | 0.6501 |
| c8aa09d | baseline look tuned on quiet frames 0–60 (dimmer 3-strand bundle, stronger outer glow), activity-driven visibility option (off), refit grid; docs/renderer.md | 0.4176 | 0.6276 | 0.6451 | 0.9139 | 0.6511 |
| (this) | look/structure: crisp 3-strand baseline, 8 packet strands (invisible when idle), 4-strand near-baseline rope, narrower core + pale-dim core gamma, black crush. **As committed (stale packets/gain): 0.5338.** Numbers here are after re-running `src.fit.coarse` (packets/gain not committed); old look under the same refit: 0.6520 | 0.4200 | 0.6512 | 0.6556 | 0.9065 | 0.6583 |
