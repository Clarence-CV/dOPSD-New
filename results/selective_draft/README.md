# Selective Draft OPSD — experiment index

First real-model run designed 2026-10-14: see `math_prelim.md` (design, budget, checklist, results). This file is the index: one row per real-model run with
job ID, config/snapshot hashes, split, and the result file. Numbers from CPU fake-model tests are not
results and never go here.

| date | job | stage | partition | config / snapshot | split, n | elapsed / SU | outputs | headline |
|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | 3510902 | full pipeline (v1) | gpu-a100-small | 512 tok, threshold | – | cancelled while pending, 0 SU | – | superseded by v2 (THU protocol) |
| 2026-10-14 | 3511305 | full pipeline (v2) | gpu-a100-small | fixed budget 128 | – | cancelled while pending, 0 SU | – | superseded by v3 (fixed threshold) |
| 2026-10-14 | 3511314 | full pipeline (v3) | gpu-a100-small | fixed threshold 0.9; rollout T=0.9 retry ≤8; 256 tok | train 80 / analysis 64 / MATH-500 test 200 (thr 0.7/0.8/0.9) | est. 4.3 h, ≤8.6 SU | `$SCRATCH/dopsd/selective_draft/math_prelim_v3` | – |

Definitions of the diagnostics (PSA, PMI, regret, induced-commit fidelity, oracle ceiling) are in
`doc/Selective_Draft_OPSD_Review.md` §3–§4.
