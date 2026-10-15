# Selective Draft OPSD — experiment index

First real-model run designed 2026-10-14: see `math_prelim.md` (design, budget, checklist, results). This file is the index: one row per real-model run with
job ID, config/snapshot hashes, split, and the result file. Numbers from CPU fake-model tests are not
results and never go here.

| date | job | stage | partition | config / snapshot | split, n | elapsed / SU | outputs | headline |
|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | 3510902 | full pipeline (v1) | gpu-a100-small | 512 tok, threshold | – | cancelled while pending, 0 SU | – | superseded by v2 (THU protocol) |
| 2026-10-14 | 3511305 | full pipeline (v2, THU protocol) | gpu-a100-small | llada_math_threshold.json (fixed 128 / 256 tok, rollout T=0.9 pass@8) | train 80 / analysis 64 / MATH-500 test 200 | est. 5.1 h, ≤10.1 SU | `$SCRATCH/dopsd/selective_draft/math_prelim_v2` | – |

Definitions of the diagnostics (PSA, PMI, regret, induced-commit fidelity, oracle ceiling) are in
`doc/Selective_Draft_OPSD_Review.md` §3–§4.
