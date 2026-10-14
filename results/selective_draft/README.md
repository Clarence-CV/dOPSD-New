# Selective Draft OPSD — experiment index

First real-model run designed 2026-10-14: see `math_prelim_v1.md` (design, budget, checklist, results). This file is the index: one row per real-model run with
job ID, config/snapshot hashes, split, and the result file. Numbers from CPU fake-model tests are not
results and never go here.

| date | job | stage | partition | config / snapshot | split, n | elapsed / SU | outputs | headline |
|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | (pending) | full pipeline | gpu-a100-small | llada_math_threshold.json | MATH pool: train 80 / analysis 64 / dev 64 | est. 5.2 h, ≤10.9 SU | `$SCRATCH/dopsd/selective_draft/math_prelim_v1` | – |

Definitions of the diagnostics (PSA, PMI, regret, induced-commit fidelity, oracle ceiling) are in
`doc/Selective_Draft_OPSD_Review.md` §3–§4.
