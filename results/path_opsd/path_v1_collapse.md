# path_v1 (Vista job 1128053) — training collapse analysis

Run: Hybrid Path OPSD basic version, branch path-opsd, code 3f792f1, `configs/llada2_mini_path.json`
(teacher = current model, tau 0.5, T 0.9, pass@8 first-correct; capped CE 0.95 on every committed token of the path
before EOS; LoRA r32, lr 2e-5; **no anchor, no LOO**). Started 2026-11-09 13:23, cancelled 22:49 at optimizer step
246/500. Cost **9.44 SU**. No eval ran. Checkpoints up to step 225 remain in `$SCRATCH/dopsd/sibling/path_v1/train/` (Vista).

## 1. What happened (train/log_rank0.jsonl, 25-round means)

| rounds | first-attempt acc | truncated | output tokens | group size | capped CE | p(c) | KL(p0‖p) |
|---|---|---|---|---|---|---|---|
| 0–25 | 0.58 | 35% | 1185 | 7.7 | 0.20 | 0.83 | 0 |
| 75–100 | 0.44 | 43% | 1341 | 12.6 | 0.14 | 0.87 | 2.2 |
| 100–125 | 0.22 | 83% | 1754 | 14.2 | 0.09 | 0.91 | 3.8 |
| 150–175 | 0.06 | 97% | 1974 | 24.3 | 0.03 | 0.97 | 13.4 |
| 300–318 | 0.03 | 100% | 2031 | 30/32 | ~0 | ~1.0 | — |

The higher TPF (5.6 -> 7.9) is fast generation of garbage, not a speedup.

## 2. Two phases (question_attempts_rank*.jsonl, all attempts, by policy version)

| policy | attempt acc | attempts truncated | acc when ended by EOS | acc when truncated | questions solved | chosen paths that are truncated |
|---|---|---|---|---|---|---|
| 0–25 | 0.37 | 58% | 0.80 | 0.07 | 0.88 | 10% |
| 25–50 | 0.31 | 64% | 0.74 | 0.06 | 0.84 | 12% |
| 50–75 | 0.28 | 54% | 0.55 | 0.04 | 0.82 | 7% |
| 75–100 | 0.21 | 54% | 0.40 | 0.05 | 0.74 | 14% |
| 100–125 | 0.08 | 89% | 0.39 | 0.04 | 0.43 | 42% |
| 125–150 | 0.06 | 96% | 0.63 | 0.04 | 0.38 | 57% |
| 150–175 | 0.03 | 100% | — | 0.02 | 0.19 | 93% |
| 200–250 | 0.02 | 100% | — | 0.02 | 0.17 | 100% |

**Phase 1 (0–100): self-sharpening drift.** Only ~10% of the chosen paths are truncated, yet accuracy on rollouts that
end normally already drops from 0.80 to 0.40 while the group size grows from 7.7 to 12.6. The loop:

1. The targets are the model's **own sampled tokens** (T 0.9, kept when p >= 0.5), pushed to p >= 0.95. That is entropy
   minimisation on its own samples.
2. The teacher is the **current** model at a **fixed** tau of 0.5. Once confidence rises, more tokens pass 0.5, so the
   next teacher path commits larger, less coordinated groups and has lower quality.
3. Those paths become the next targets. Nothing pulls the model back: anchor_weight = 0, and KL(p0‖p) is only logged
   (0 -> 2.2 by step 90).
4. The only brake is the answer-level correctness filter. It certifies the final answer, not each parallel group, so
   glitchy groups on a path that still reaches the right answer get sharpened too.

The stable sibling run (sib_t05_v1) differed on exactly these points: frozen base teacher, LOO-capped target and a
0.1 KL anchor.

**Phase 2 (100+): truncated paths take over.** `path_steps` drops only tokens at or after EOS. A truncated rollout has no
EOS, so **its whole looping tail is supervised**. The tail also holds most of the steps, so 8 random states per path
land mostly inside the loop. Repetition loops are the lowest-entropy text, so capped CE pushes them to 0.95 easily and
commits whole blocks (group size 30/32). More loops -> more truncation -> fewer EOS-correct paths -> the surviving
"correct" paths are almost all truncated.

**Why truncated outputs still pass the grader.** The vendored DMax scorer
(`selective_draft/vendor/dmax_val_math.py: extract_llm_final_answer_candidates`) is lenient by design: candidates come
from a pattern near the end, the last `\boxed{}` (only if no substantial suffix), the last LaTeX block in the last 800
characters, the last 3 lines and the right-hand side of `=`. A loop that repeats an intermediate or final value can
match. Indirect evidence: among questions solved late (policy 150–250), 61% have a 1–2 digit integer gold vs 43% among
unsolved ones (early solved: 50%). This points to false positives but does not prove them. Generation text is not
saved in the logs, so this was not confirmed directly; one gh-dev decode of checkpoint-step-225 would settle it.

## 3. Root causes (ranked)

1. **Positive feedback without a fixed reference**: self-generated targets + current-model teacher at a fixed tau + no
   anchor. This alone explains phase 1.
2. **Truncated rollouts accepted as teacher paths, with the no-EOS tail fully supervised.** This is the amplifier
   behind phase 2 and a plain bug relative to the intent ("EOS / post-EOS excluded" assumed an EOS exists).
3. **Lenient verifier** lets degenerate outputs count as correct, so the filter stops filtering exactly when it is
   needed.
4. **No collapse guard**: nothing stopped the run when truncation or KL exploded. About 8 of the 9.44 SU were spent
   after step 100.

The dev job 1127965 (4 updates) could not see any of this. Phase 1 needs ~50+ updates to show.

## 4. Fixes

Implemented 2026-11-10 (branch path-opsd, uncommitted at the time of writing):
- `sibling.path_teacher = frozen` (default): the pretrained model samples the paths with seeds independent of the
  training step -> the targets are a fixed dataset, no self-sharpening loop (`current` kept as the path_v1 control).
- A path is accepted only if verified correct AND `stop_reason == eos`.
- `cli collect-paths` + pipeline stage `paths`: paths sampled once, stored compactly, reused by `train`
  (`sibling.path_source`); the trainer iterates only over questions with a path.
Still open: a stricter training grader (last boxed -> Math-Verify, no tail fallback; see doc/lit_graders_dllm_rl.md),
trajectory filters (length ratio, repetition), the LOO / marginalized target for stage 2.

Original options list (kept for the record):

- **Required for any rerun**: accept only `stop_reason == "eos"` paths (or truncate supervision at the end of the
  answer); fail fast when 50-round truncation > 0.6 or KL(p0‖p) > ~1.
- **Break the feedback loop** (pick one or two): frozen base teacher (as in the sibling run); a KL(p0‖p) anchor
  in the loss; refresh the teacher policy only every N updates.
- **Stricter verifier for training paths**: require EOS and a final `\boxed{}` with no suffix.
- Run a cheap ~100-update probe (gh-dev or short gh) watching EOS-accuracy, truncation and KL before the next main run.
