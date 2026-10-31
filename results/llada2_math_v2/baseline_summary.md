# LLaDA2.0-mini MATH-500 baseline (job 3526876, DMax-aligned)

Config: block 32, tau 0.95 (>= min(tau, best-1e-5)), T=0, cap 2048, DMax prompt and scorer, no KV cache, code 3b92acb.
Job: gpu-a100, 3 GPUs, TIMEOUT at 10:00:09, 30.0 SU. 494/500 rows (shard 0 stopped at 161/167; missing test-483/486/489/492/495/498).
Full rows: `$WORK/inf385t/results/llada2/llada2_math_v2/eval/base_test_dmax_boxed/` (backed up 2026-10-31).

| metric | value | DMax |
|---|---|---|
| accuracy | 375/494 = 75.9% | 75.8% |
| TPF, mean of rows / pooled | 2.88 / 2.47 | 2.58 |
| mean NFE (median) | 394 (231) | |
| mean output tokens | 974 | |
| latency mean / median / p90 | 202 / 108 / 520 s | |
| stopped by EOS / hit 2048 cap | 370 / 124 (25.1%) | |
| accuracy when EOS / when capped | 97.6% / 11.3% | |
| seconds per forward | 0.53 | |

Latency: correct rows mean 117 s, wrong rows mean 471 s (wrong rows are mostly capped).
Final-layer commits: 69% of steps commit 1 token, 31% commit >= 2 (those carry 73% of all committed tokens); 53% of steps have no token above 0.95 (forced best-one commit).
