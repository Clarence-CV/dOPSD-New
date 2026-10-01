# Experiment log: THU d-OPSD reproduction on TACC Lonestar6

Allocation CDA26008 (class-shared). SU = nodes × max(elapsed h, 0.25) × rate. Request walltimes tightly (~1.3× the expected runtime): shorter requests backfill more easily. Rates (fetched 2026-09-20, CLAUDE.md B.3):
gpu-a100 / gpu-a100-dev 3 SU/h (3× A100-40GB), gpu-a100-small 1.5 SU/h (1× A100), gpu-h100 6 SU/h.
Elapsed times come from `sacct`. Paths: code `$WORK/inf385t/dOPSD`, outputs `$SCRATCH/dopsd/`.

## Running total

| as of | SU spent |
|---|---|
| 2026-10-01 | 22.08 |

## Jobs

| date | job | partition | purpose | config | elapsed | SU | status | outputs | key result |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-27 | 3474357 | gpu-a100-dev (idev) | environment + pipeline debugging | dev_smoke.sh stages env / eval_trace / train (16 steps) / resume (to 24) / eval_adapter | 01:56:55 | 5.85 | done (scancel at end) | `dopsd/dev/` | all 5 stages pass; base LLaDA 6/8 GSM8K; A100-40GB OOMs without LLaDA activation ckpt; ~8.9 s/step on 3 GPUs; full ckpt 2.7 GB, bf16 adapter 321 MB |
| 2026-09-28 | 3474955 | gpu-a100-small | analysis pipeline smoke v1 | base, GSM8K 20 problems, greedy, fixed 128 & 64 steps, trace + pi_sibling v1 | 00:33:47 | 0.84 | done | `dopsd/anasmoke/s{128,64}/{traces,pisib}` | pipeline OK; 16 GB peak; base acc 75% (128) / 80% (64) |
| 2026-09-28 | 3476726 | gpu-a100-small | analysis v2 (coord gap, controls) | same traces, s128, batch 16 | 00:27:16 | 0.68 | done | `dopsd/anasmoke/s128/pisib2` | ~75 s/rollout; 16.9 GB peak |
| 2026-09-28 | 3476727 | gpu-a100-small | analysis v2 | same traces, s64 | 00:22:17 | 0.56 | done | `dopsd/anasmoke/s64/pisib2` | 17.0 GB peak |
| 2026-09-28 | 3476761 | gpu-a100-small | threshold-decoding smoke | base, 20 problems, greedy, threshold 0.9, trace + analysis v2 | 00:28:12 | 0.71 | done | `dopsd/anasmoke/thr0.9/` | acc 75%; ~85 effective steps/rollout; wide |C_t| distribution |
| 2026-09-30 | 3474952 | gpu-a100 | **main training** | GSM8K d-OPSD, 3 GPUs, BATCH_DIVIDE 8, 1344 steps, ACT_CKPT whole_layer, adapters every 64, full ckpt every 128 (keep 1), trace every 25 rounds | 03:41:00 (09:06-12:47, after ~57 h in queue) | 11.05 | done (exit 0) | `dopsd/runs/gsm_opsd` → `$WORK/inf385t/runs/gsm_opsd` (21 adapters, checkpoint-1344, 21 traces) | train_runtime 12699 s; loss can be negative (d-OPSD clamps per-vocab-entry KL terms via jsd_token_clip); grad_norm 0 on wrong rollouts (loss*0 by design) |
| 2026-09-30 | 3479225 | gpu-a100-small | analysis v3 (order metrics), s128 smoke traces | bf16 re-scoring | 00:24:35 | 0.61 | done | `dopsd/anasmoke/s128/pisib3` | order metrics OK (17.1 GB peak); sanity rank_S <= |C_t| fails for 2.3% of tokens (near-ties, 97/105 off by one rank) -> the sampler runs fp16 autocast, so the analysis now defaults to fp16 |
| 2026-09-30 | 3479226 | gpu-a100-small | analysis v3, s64 smoke traces | bf16 re-scoring | 00:22:21 | 0.56 | done | `dopsd/anasmoke/s64/pisib3` | 17.1 GB peak |
| 2026-09-30 | 3479227 | gpu-a100-small | analysis v3, thr0.9 smoke traces | bf16 re-scoring | 00:21:51 | 0.55 | done | `dopsd/anasmoke/thr0.9/pisib3` | 17.1 GB peak |
| 2026-09-30 | 3481013 | gpu-a100-small | fp16 re-scoring check, s128 smoke traces | --autocast_dtype float16 (new default) | 00:27:21 | 0.68 | done | `dopsd/anasmoke/s128/pisib4` | rank_S > |C_t| violations: bf16 105/4480 (2.34%) -> fp16 2/4480 (0.04%, both off by one rank); 18.2 GB peak. The fp16 default is confirmed |
| 2026-09-30 | 3480928 | gpu-a100 | base model, full GSM8K | – | never ran | 0 | cancelled (we look for phenomena, not exact paper numbers; base on the 300 subset suffices) | – | – |
| 2026-09-30 | 3480929 | gpu-a100 | adapter sweep (all 21 snapshots) | – | never ran | 0 | cancelled; replaced by 3480947 | – | – |
| 2026-09-30 | 3480947 | gpu-a100 | adapter sweep (13 ckpts) | – | never ran | 0 | cancelled; too many checkpoints for a phenomenon study | – | – |
| 2026-09-30 | 3480958 | gpu-a100 | adapter sweep (4 ckpts) | – | never ran | 0 | cancelled; merged into the pipeline job below (one queue wait instead of two) | – | – |
| 2026-09-30 | 3480959 | gpu-a100 | pipeline v1 (fixed budget only) | – | never ran | 0 | cancelled; replaced by the 2x2 version | – | – |
| 2026-09-30 | 3480966 | gpu-a100 | **pipeline** (one job, 12.5 h request) | (1) sweep, greedy + fixed 128: base + steps 448/576/1152/1344 on 300 seeded problems; (2) pick best adapter; (3) traced rollouts, T=0.9, 300 problems, for {base, best} x {fixed budget 128, threshold 0.9}; (4) PI/sibling/order analysis, 3 shards each. Figures/report made offline | – | est. ~27 | pending | `dopsd/pipeline_gsm_opsd/` (csv/txt/log copied to `$WORK/inf385t/results/pipeline_gsm_opsd/`) | – |

## Result snapshots

Per-analysis reports (tables + definitions) live next to the figures as `report.md`, with raw tables
`tokens.csv.gz`, `steps.csv.gz`, `rollouts.csv`, `summary.csv`.

| date | what | report |
|---|---|---|
| 2026-09-30 | smoke (base, 20 problems, greedy, bf16 re-scoring): fixed 128 / 64 + threshold 0.9, all metrics incl. order / effect types / distance-matched co-gain | `results/smoke_20260929/report.md` (token/step tables in `outputs/smoke_20260929/`) |
