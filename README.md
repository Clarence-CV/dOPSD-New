# On-Policy Self-Distillation for Diffusion LLMs (dLLM)

<p align="center">
<a href="https://arxiv.org/abs/2607.04428"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
</p>

---
## Overview

This repository adapts **On-Policy Self-Distillation (OPSD)** to **masked diffusion
language models** (dLLMs) such as [Dream-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)
and [LLaDA-8B](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

A single model plays both **student** and **teacher**. The student conditions on a
*noisy / partially-masked* view of its own generation; the teacher conditions on
*privileged information* (the ground-truth solution, or the concrete final
rollout). We then perform token-level distribution matching at the masked
positions, along the student's **own on-policy decoding path** — so the
supervision is drawn from states the model actually visits at inference rather
than from externally-injected noise.

Two on-policy variants are provided:

- **Remask OPSD** (`opsd_dllm_train.py`) — the student rolls out a completion,
  a subset of its positions is re-masked, and the teacher (optionally seeing the
  reference solution) supervises the masked tokens.
- **Trajectory OPSD** (`opsd_dllm_trajectory_train.py`) — the student's noisy view
  is a **real intermediate decoding step** (a partial state from the diffusion
  trajectory), and the teacher sees the **concrete final rollout**. This keeps
  the supervision exactly on the model's confidence-ordered decoding schedule.

## Installation

```bash
conda env create -f environment.yaml
conda activate opsd
```

Dream-7B is run with `sdpa` attention (it is **not** safe with
`flash_attention_2`), so a FlashAttention install is not required.

The trainer builds on `trl`'s experimental GOLD/distillation machinery.

## Repository Structure

```
├── opsd_dllm_trainer.py              # Remask OPSD trainer (Dream / LLaDA)
├── opsd_dllm_train.py                # Remask OPSD entry point
├── data_collator_dllm.py             # Collator for remask OPSD
├── opsd_dllm_trajectory_trainer.py   # Trajectory OPSD trainer (on decoding-path)
├── opsd_dllm_trajectory_train.py     # Trajectory OPSD entry point
├── data_collator_dllm_trajectory.py  # Collator for trajectory OPSD
├── sft_train_dllm.py                 # SFT baseline (masked-prediction loss)
├── llada_utils.py                    # LLaDA sampling / masking helpers
├── accelerate.yaml                   # Accelerate config (multi-GPU)
├── scripts/
│   ├── run_opsd_dllm_7b.sh           # Launch remask OPSD (Dream-7B / LLaDA-8B)
│   ├── run_opsd_dllm_trajectory_7b.sh# Launch trajectory OPSD
│   └── run_sft_dllm.sh               # Launch SFT baseline
└── eval/
    ├── Dream/                        # Dream eval (lm-eval harness + LoRA merge)
    │   ├── eval_instruct/eval.sh     # gsm8k / math500 / humaneval / mbpp
    │   └── merge_lora.py             # Merge a LoRA adapter into the base model
    ├── LLaDA/                        # LLaDA eval
    └── run_eval_dllm.sh              # Quick GSM8K Avg@1 launcher
```

## Quick Start

### 1. Train

```bash
# Trajectory OPSD on Dream-7B (recommended; supervision stays on the decoding path)
bash scripts/run_opsd_dllm_trajectory_7b.sh

# Remask OPSD on Dream-7B
bash scripts/run_opsd_dllm_7b.sh
```

Both scripts default to the Dream-7B backend; switch to LLaDA-8B with
`STUDENT_BACKEND=llada`, and select GPUs with `GPU_IDS=0,1,2,3`. Checkpoints are
LoRA adapters written to `outputs/opsd_dllm[_trajectory]/<run_config>/checkpoint-<step>/`.

### 2. Merge the LoRA adapter

Evaluation loads a full model, so merge the trained adapter into the base model
first (edit the checkpoint / output paths at the top of the script):

```bash
python eval/Dream/merge_lora.py
```

### 3. Evaluate

```bash
# Full instruct benchmarks via the lm-eval harness
bash eval/Dream/eval_instruct/eval.sh      # gsm8k_cot, minerva_math500, humaneval_instruct, mbpp_instruct

# or a quick single-task GSM8K Avg@1
bash eval/run_eval_dllm.sh
```

> **Train/eval consistency (hard invariant).** The diffusion sampler must use
> `steps == max_new_tokens` (`--gen_steps == --gen_max_new_tokens` in training,
> `--diffusion_steps == --max_new_tokens` in eval). Fewer steps than tokens
> forces multiple commits per step and collapses quality. Keep
> `max_prompt_length + max_new_tokens` under Dream-v0's 2048-position limit.

## Datasets & Models

| Toggle | Options |
|---|---|
| `STUDENT_BACKEND` | `dream` → `Dream-org/Dream-v0-Instruct-7B` (default) · `llada` → `GSAI-ML/LLaDA-8B-Instruct` |
| Dataset | `horseee/MixChain-Z-PRM12K` (question → problem, answer → solution) |

## Training details

### Remask OPSD — `scripts/run_opsd_dllm_7b.sh`

| Argument / env | Description |
|---|---|
| `--beta` | JSD mixture weight. `0` = forward KL (stable default for on-policy remask), `1` = reverse KL. |
| `MASK_SCHEDULE` | `diffusion` (per-example antithetic rate, i.i.d. Bernoulli masking) or `fixed` (exact count `k = round(n·ratio)`; set via `FIXED_MASK_RATIO`). |
| `USE_PI` | `1` = teacher prompt embeds the reference solution (privileged information); `0` = no-PI baseline (teacher prompt == student prompt). |
| `OFF_POLICY` | `0` = on-policy: student rolls out, then re-mask. `1` = off-policy: distill on the dataset's ground-truth answer (no rollout). |
| `--gen_*` | On-policy rollout controls: `--gen_max_new_tokens`, `--gen_steps`, `--gen_temperature`, `--gen_top_p`, `--gen_alg` (Dream-only). |
| `--fixed_teacher` | Freeze the teacher to the initial policy via a LoRA adapter (main setting). |
| `--jsd_token_clip` | Per-token JSD clip; caps stylistic tokens that otherwise dominate the loss. |

### Trajectory OPSD — `scripts/run_opsd_dllm_trajectory_7b.sh`

| Argument / env | Description |
|---|---|
| `TRAJ_MASK_THRESHOLD` | Step-eligibility cutoff — only decoding steps whose masked fraction exceeds this are considered as the student's noisy view. |
| `TRAJ_STEP_SELECT` | Which eligible step to take: `least` (closest to the threshold), `most` (noisiest), or `random`. |
| `TRAJ_TEACHER_VIEW` / `TRAJ_TEACHER_GAP` | Teacher target: `snapshot` (single forward; `gap = -1` → concrete final rollout, `n ≥ 0` → peek `n` steps ahead) or `all_future` (average over remaining steps). |
| `--filter_wrong_rollouts` | Verify-gated: rollouts whose final answer is wrong receive no teacher signal (loss 0 there), so only correct on-policy trajectories are distilled. |

### SFT baseline — `scripts/run_sft_dllm.sh`

Vanilla masked-prediction SFT for the diffusion LM (`sft_train_dllm.py`), sharing
the same `MASK_SCHEDULE` toggles as remask OPSD for a controlled comparison.

## Citation

If you find this useful, please consider citing:
```bibtex
@misc{dat2026dopsdonpolicyselfdistillationdiffusion,
      title={dOPSD: On-Policy Self-Distillation for Diffusion Language Models}, 
      author={Phuong Tuan Dat and Qi Li and Xinchao Wang},
      year={2026},
      eprint={2607.04428},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.04428}, 
}
```

## Acknowledgements

This work builds on On-Policy Self-Distillation ([OPSD](https://arxiv.org/pdf/2601.18734v3))
