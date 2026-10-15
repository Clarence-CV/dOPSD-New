# selective_draft — Selective Draft OPSD pipeline

Method: `doc/Selective_Draft_OPSD.md`. Plan: `doc/Selective_Draft_OPSD_Engineering_Plan.md`.
Review that motivated the candidate teacher defaults below: `doc/Selective_Draft_OPSD_Review.md`.

```
hidden after block[early_layer-1] -> EarlyReadoutAdapter g -> frozen final norm + LM head -> r_i
pick A = argmax_i max r_i over the current block's MASKs; gate = max r_A >= message_threshold (0.8)
m_A = sum_v r_A(v) E(v);  h_A += MessageInjector J(h_A, m_A)   (only A, only when gated)
native tail layers -> final logits -> threshold commit (> commit_threshold, else top-1) or fixed budget
```

| module | role |
|---|---|
| `config.py` | dataclasses, JSON, `--set a.b=v` overrides, validation |
| `backends/` | LLaDA (checked against rev 08b83a6f) / iLLaDA (static check only) module paths, `project()` = norm + head |
| `modules.py` | `EarlyReadoutAdapter` (identity at init), `MessageInjector` (zero at init), `soft_message` |
| `model.py` | `GuidedModel`: scoped hook on the boundary block; modes `off` / `readout_only` / `single_soft`; `GuidanceSpec` for training and the oracle patch |
| `generation.py` | one decoder for collect / rollouts / eval; threshold + fixed budget; left padding; `block_end` stop |
| `trace.py` | CPU `StateRecorder` (state_only / diagnostic), shards, `replay_check` |
| `teacher.py` | reveal-A targets: `realized` (default) / `argmax` / `mc` / `topk`; `oracle_patch_log_probs` |
| `losses.py` | full-vocab forward KL |
| `training.py` | `ReadoutTrainer` (stage 1), `GuidanceTrainer` (stage 2, on-policy refresh), resume, manual grad all-reduce |
| `checkpoint.py` | snapshot (`modules.safetensors` + manifest) and full checkpoint (optimizer, RNG, trainer state) |
| `evaluation.py` | accuracy + latency, commit-threshold sweep |
| `diagnostics.py` | `readout`, `teacher` (target collapse + oracle ceiling), `consistency` (PSA, PMI, regret, induced-commit fidelity) |
| `testing.py` | CPU fake with the LLaDA layout; `configs/selective_draft/fake_cpu.json` |

## Commands (real model: TACC compute node only)

```bash
C=configs/selective_draft/llada_threshold.json
python -m selective_draft.cli prepare-data --config $C --output-dir outputs/sd/data          # once; login node ok
S="--set data.split_manifest=\"outputs/sd/data/split_manifest.json\""
python -m selective_draft.cli inspect  --config $C $S
python -m selective_draft.cli collect --config $C $S --set data.split='"train"' --set guidance.mode='"off"' --limit 50 --output-dir outputs/sd/train_states
python -m selective_draft.cli collect --config $C $S --set data.split='"analysis"' --set guidance.mode='"off"' --limit 50 --output-dir outputs/sd/analysis_states
python -m selective_draft.cli train-readout --config $C $S --set data.split='"train"' --states outputs/sd/train_states --max-steps 200 --output-dir outputs/sd/readout
python -m selective_draft.cli diagnose --kind readout --config $C $S --snapshot outputs/sd/readout/snapshot --states outputs/sd/analysis_states --output-dir outputs/sd/readout_diag
python -m selective_draft.cli diagnose --kind teacher --config $C $S --snapshot outputs/sd/readout/snapshot --states outputs/sd/analysis_states --output-dir outputs/sd/teacher_diag
python -m selective_draft.cli train-guidance --config $C $S --set data.split='"train"' --readout-snapshot outputs/sd/readout/snapshot --max-steps 50 --output-dir outputs/sd/guidance
python -m selective_draft.cli evaluate --config $C $S --set data.split='"dev"' --set guidance.mode='"single_soft"' --snapshot outputs/sd/guidance/snapshot --output-dir outputs/sd/eval_guided
python -m selective_draft.cli diagnose --kind consistency --config $C $S --states outputs/sd/eval_guided --output-dir outputs/sd/cons
```

Staged real-model smoke: `bash tacc/selective_draft_smoke.sh <stage>` inside `idev`; batch: `tacc/selective_draft.slurm`.

## CPU tests (no pretrained model)

```bash
for t in model generation teacher_losses training regressions; do python tests/test_selective_draft_$t.py; done
python -m selective_draft.cli collect --config configs/selective_draft/fake_cpu.json --output-dir /tmp/sd/base   # fake dry run
```

## Defaults that differ from the engineering plan (reasoned in the review doc)

- `teacher.mode = realized` (A revealed as the token it actually receives; one Teacher forward). The plan's
  MC mixture is kept as `mc` / `topk` for the collapse diagnostic (`diagnose --kind teacher`).
- `evaluate` sweeps `evaluation.commit_thresholds` (accuracy–NFE curve) instead of a single threshold.
- `diagnose --kind teacher` also reports the activation-patch oracle ceiling; `--kind consistency` is new.
- `generation.forbid_mask_candidate = true`: MASK cannot be selected, but candidate confidence is its
  probability in the **original** softmax. Removing MASK must not inflate the commit confidence.

## Teacher modes

`teacher.mode` changes only the training target. Student messages remain full-soft in every mode.

| mode | Token(s) revealed at A in the Teacher input |
|---|---|
| `realized` (default candidate) | A's eventual token from the current Student rollout, including a later step if A is not committed now |
| `argmax` | The highest-probability token in the current early draft r_A |
| `mc` / `topk` (mixture approximations) | Multiple candidates from r_A; average Teacher **probabilities**, using sampling frequencies or renormalised top-K weights |

`realized` can provide future-derived identity at a sibling position. Only A is revealed; no other
future tokens are inserted. It does not guarantee A is correct or that Student can infer its future
value. Its superiority to the other targets remains an experimental question.

## Resume, measurements and diagnostics

- CUDA processes bind to `LOCAL_RANK` before loading weights. Manual gradient all-reduce includes initial
  module broadcast and empty-rank handling. Rank 0 publishes the shared checkpoint after collecting each
  rank's trainer, sampler and RNG state. Exact resume requires the same world size and dataset ordering.
- Old checkpoints without sampler state cannot provide exact resume. Their `snapshot/` remains usable
  as initialisation for a fresh run (`--snapshot` / `--readout-snapshot`), with a new optimizer.
- Latency is the median of identical-RNG timing repeats, measured without trace recording. Trace collection
  is a separate replay with the same RNG. `latency_s` on a sample row is its **batch's** latency; summaries
  count each `batch_id` once, including for NFE and gate counters. Quality uses one trajectory per example.
- Induced-commit fidelity forks the recorded pre-step state with guidance off and continues until B really
  commits; it does not condition on the guided sibling decisions. This compares with the base policy, not
  ground truth. It currently supports greedy threshold traces with recorded raw-confidence protocol metadata.
  Sampling, fixed-budget and legacy traces report an explicit unsupported/missing-metadata status.
- The re-mask regret and reveal-response (`pmi` legacy key) metrics are model-consistency diagnostics;
  neither is a correctness certificate. Activation patch recovery is an intervention result, not a proven
  upper bound on what a learned injector can achieve.
- The smoke script collects separate `train_states` and `analysis_states`; readout diagnostics do not reuse
  the training states. No real-model or GPU-distributed validation is implied by CPU regression tests.

## MATH preliminary pipeline (one job)

`tacc/selective_draft_math.slurm` runs data → parity smoke → collection → stage 1 → training-free diagnostics →
stage 2 → guided states → timed eval → `REPORT.md` (`python -m selective_draft.report RUN_DIR`). Submit from WSL with
`bash tacc/sync_and_submit_math.sh`. Design and budget: `results/selective_draft/math_prelim.md`.
`data.task=math` uses `ankner/math-500`, the legacy MATH prompt, and `eval/parser_helper.is_equiv`.
