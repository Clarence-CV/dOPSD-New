# selective_draft — Selective Draft OPSD pipeline

Method: `doc/Selective_Draft_OPSD.md`. Plan: `doc/Selective_Draft_OPSD_Engineering_Plan.md`.
Review that fixed the defaults below: `doc/Selective_Draft_OPSD_Review.md`.

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
python -m selective_draft.cli collect  --config $C $S --set guidance.mode='"off"' --limit 50 --output-dir outputs/sd/base_states
python -m selective_draft.cli train-readout --config $C $S --states outputs/sd/base_states --max-steps 200 --output-dir outputs/sd/readout
python -m selective_draft.cli diagnose --kind readout --config $C $S --snapshot outputs/sd/readout/snapshot --states outputs/sd/base_states --output-dir outputs/sd/readout_diag
python -m selective_draft.cli diagnose --kind teacher --config $C $S --snapshot outputs/sd/readout/snapshot --states outputs/sd/base_states --output-dir outputs/sd/teacher_diag
python -m selective_draft.cli train-guidance --config $C $S --set data.split='"train"' --readout-snapshot outputs/sd/readout/snapshot --max-steps 50 --output-dir outputs/sd/guidance
python -m selective_draft.cli evaluate --config $C $S --set data.split='"dev"' --set guidance.mode='"single_soft"' --snapshot outputs/sd/guidance/snapshot --output-dir outputs/sd/eval_guided
python -m selective_draft.cli diagnose --kind consistency --config $C $S --states outputs/sd/eval_guided --output-dir outputs/sd/cons
```

Staged real-model smoke: `bash tacc/selective_draft_smoke.sh <stage>` inside `idev`; batch: `tacc/selective_draft.slurm`.

## CPU tests (no pretrained model)

```bash
for t in model generation teacher_losses training; do python tests/test_selective_draft_$t.py; done
python -m selective_draft.cli collect --config configs/selective_draft/fake_cpu.json --output-dir /tmp/sd/base   # fake dry run
```

## Defaults that differ from the engineering plan (reasoned in the review doc)

- `teacher.mode = realized` (A revealed as the token it actually receives; one Teacher forward). The plan's
  MC mixture is kept as `mc` / `topk` for the collapse diagnostic (`diagnose --kind teacher`).
- `evaluate` sweeps `evaluation.commit_thresholds` (accuracy–NFE curve) instead of a single threshold.
- `diagnose --kind teacher` also reports the activation-patch oracle ceiling; `--kind consistency` is new.
- `generation.forbid_mask_candidate = true`: the MASK id is removed from the candidate softmax for base and
  guided alike (a sampled MASK would otherwise be an invalid commit).
