# selective_draft — Selective Draft OPSD pipeline

Research overview, method history and mathematical caveats: [Selective_Draft_OPSD.md](../doc/Selective_Draft_OPSD.md).
LLaDA2.0-mini work: [handoff](../doc/llada2-mini_handoff.md) and the section "LLaDA2.0-mini" at the end of this file.

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
| `diagnostics.py` | `readout`, `teacher` (target signal + activation-patch recovery), `consistency` (PSA, reveal response / legacy `pmi`, regret, induced-commit fidelity) |
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

## Single-source defaults and diagnostics

- `teacher.mode = realized` (A revealed as the token it actually receives; one Teacher forward).
  `mc` / `topk` remain available to measure mixture signal (`diagnose --kind teacher`); see the overview §4.
- `evaluate` sweeps `evaluation.commit_thresholds` (accuracy–NFE curve) instead of a single threshold.
- `diagnose --kind teacher` also reports activation-patch recovery (not a proven upper bound); `--kind consistency` reports coordination diagnostics.
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

## Joint-reference OPSD（分支 `joint`）

多位置同步 soft 消息、pass@8 同状态续写、leave-one-out Teacher 认证与共同参考蒸馏，见 [JOINT.md](JOINT.md)。入口是 `train-joint`，默认配置为 `configs/selective_draft/llada_joint_math.json`；原单 Anchor 流程保持可用。

## LLaDA2.0-mini（`backend=llada2_mini`，2026-10-19）

配置 `configs/selective_draft/llada2_mini_math.json`；一个可续跑的作业 `tacc/llada2_pipeline.slurm`（`STAGES` 选阶段，`SMOKE=1` 全阶段小规模走一遍）；本地同步提交 `tacc/sync_and_submit_llada2.sh`；登录节点一次性环境 `tacc/setup_llada2_env.sh`（conda env `llada2` = dOPSD + transformers 4.57.1，下载固定 revision 的权重和数据，并离线自检）。

| 组件 | 位置 | 要点 |
|---|---|---|
| backend | `backends/llada2_mini.py` | `AutoModelForCausalLM` 加载；按绝对位置构造 32 格 block-causal 4D mask；batch 1、无 padding；所有 MoE 层换成去掉 `no_grad` 的等价 `moe_infer`（no_grad 下逐位相同，测试覆盖）|
| 解码 | `block_generation.py` | block 从位置 0 起算，prompt 尾可与首个生成 block 共用；每步 conf>τ 全提交，否则 top-1；T=0 为 argmax（修正 HF 2.0 的 multinomial）；`prefix_complete_eos`：每次提交后检查，首个 EOS 左侧无 MASK 即停；cap 外 EOS 忽略；只对当前 block 做 softmax |
| 组规则 | `model.py::_select_group_threshold` | `guidance.group_policy=confidence_threshold_all`：早层 readout 候选的原始概率 > 本步解码 τ（经 `StepContext.commit_threshold` 传入，不是第二个参数）的位置全部入组；<2 不通信；末 block 向上取整的尾部不入组 |
| prompt / 评分 | `data.py`、`scoring.py`、`klass_math.py` | `klass_final_answer`（KLASS system prompt）与 `boxed_step_by_step`；`data.scorer=klass` 用 KLASS 原样代码判分，legacy 判分并列保存为交叉核对 |
| 子划分 | `data.py::SUBSPLITS` | 按 SHA256 排序切 readout_val/confirm、sampling_pilot、dev_fast 等（计划 §6.5），`data.subsplit` 选择 |
| 评测 | `evaluation.py`、`cli merge-eval` | `--shard I/N` 多卡切分、`--resume` 按 example_id 续跑、指纹防止混用；行内含 completion token、EOS/role_end 位置、物理长度 |
| 层诊断 / readout 选点 | `layer_profile.py`、`cli layer-profile` / `readout-sweep` | 计划 §7.1 的稳定过阈层规则（1-SE 取最早层）→ `layer_decision.json`；§7.1.2 门槛与选点 → `readout_sweep.json` |
| Joint 训练 | `root_joint.py`（`training.sampling_protocol=root_success_v2`）| 每 rank 每轮一道新题，最多 K 次从 prompt 采样、首次答对即停；成功轨迹均匀抽 1 个有组状态，用自身最终 token 做全员 LOO；失败只记账（`question_attempts_rank*.jsonl`），不重抽；RNG 由 (seed, 题号, attempt) 派生，续跑精确 |

```bash
# CPU 测试（需要 transformers 4.57：本地用 hw1py39 环境）
python tests/test_selective_draft_llada2.py
# TACC：先小规模全阶段冒烟（单卡），再分阶段正式跑
SMOKE=1 STAGES="baseline states profile readout joint jointeval" SBATCH_ARGS="-p gpu-a100-small -t 00:45:00" bash tacc/sync_and_submit_llada2.sh
STAGES=baseline SBATCH_ARGS="-t 03:00:00" bash tacc/sync_and_submit_llada2.sh
```

未实现：prefix KV cache（计划 W9）、真实 mini 上的消息路径/干预审计（§11.3）、Math-Verify 交叉判分、THU/mini-CAP 对照。
