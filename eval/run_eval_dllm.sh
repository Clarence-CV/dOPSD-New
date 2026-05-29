#!/bin/bash
# GSM8K evaluation for Dream-7B (diffusion LM): vanilla base vs. SFT adapter.
# Metric: Avg@1 (single sample per problem, exact-match on the boxed final number).
#
# HARD INVARIANTS (must match training — see scripts/run_opsd_dllm_7b.sh):
#   * --diffusion_steps MUST equal --max_new_tokens. steps < tokens forces the
#     diffusion sampler to commit multiple tokens per step → quality collapse.
#   * --max_new_tokens should match training's --gen_max_new_tokens (768) plus
#     stay under Dream-v0's 2048 position limit once the prompt is added.
#   * --generator diffusion is explicit: never silently fall back to AR generate.
#   * --val_n 1 → Avg@1. Temperature is kept low (0.2) so the single sample is
#     close to greedy; raising it just adds variance to a one-shot metric.
set -euo pipefail

cd "$(dirname "$0")"          # cd into eval/ so `python evaluate_aime_dllm.py` resolves
REPO_ROOT="$(cd .. && pwd)"   # repo root, for absolute checkpoint paths

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
DATASET="${DATASET:-gsm8k}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# SFT adapter to evaluate. Override with SFT_CHECKPOINT_DIR=/abs/path/... when invoking.
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/dream7b-fix075-2epochs-30k/sft_dllm_dream7b_fix075_2epochs_30k/checkpoint-100}"

COMMON_ARGS=(
    --base_model          "$BASE_MODEL"
    --dataset             "$DATASET"
    --val_n               1
    --batch_size          2
    --max_new_tokens      768
    --diffusion_steps     768
    --temperature         0.2
    --top_p               0.95
    --alg                 entropy
    --alg_temp            0.0
    --generator           diffusion
    --torch_dtype         "$TORCH_DTYPE"
    --attn_implementation sdpa
    --device_map          "$DEVICE_MAP"
)

# --- 1) Vanilla Dream-7B (no LoRA). ---
echo "=== Eval: vanilla $BASE_MODEL on $DATASET (Avg@1) ==="
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
    "${COMMON_ARGS[@]}"

# --- 2) SFT adapter on top of Dream-7B. ---
echo "=== Eval: SFT adapter $SFT_CHECKPOINT_DIR on $DATASET (Avg@1) ==="
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint_dir      "$SFT_CHECKPOINT_DIR"
