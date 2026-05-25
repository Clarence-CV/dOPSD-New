#!/bin/bash
# AIME evaluation for Dream-7B (diffusion LM) OPSD checkpoints.
#
# HARD INVARIANTS (must match training — see scripts/run_opsd_dllm_7b.sh):
#   * --diffusion_steps MUST equal --max_new_tokens. steps < tokens forces the
#     diffusion sampler to commit multiple tokens per step → quality collapse.
#   * --max_new_tokens should match training's --gen_max_new_tokens (768) plus
#     stay under Dream-v0's 2048 position limit once the prompt is added.
#   * --generator diffusion is explicit: never silently fall back to AR generate.
set -euo pipefail

cd "$(dirname "$0")"          # cd into eval/ so `python evaluate_aime_dllm.py` resolves
REPO_ROOT="$(cd .. && pwd)"   # repo root, for absolute checkpoint paths

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
DATASET="${DATASET:-math500}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Point this at a checkpoint from the new forward-KL run. Override with
# CHECKPOINT_DIR=/abs/path/... when invoking the script.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/outputs/opsd_dllm/dream7b_gen768_forwardbeta0_v2/checkpoint-300}"

# --- Base-model baseline (no LoRA). Uncomment to measure vanilla Dream-7B. ---
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 python evaluate_aime_dllm.py \
#     --base_model          "$BASE_MODEL" \
#     --dataset             "$DATASET" \
#     --val_n               16 \
#     --batch_size          2 \
#     --max_new_tokens      768 \
#     --diffusion_steps     768 \
#     --temperature         0.5 \
#     --top_p               0.95 \
#     --alg                 entropy \
#     --alg_temp            0.5 \
#     --generator           diffusion \
#     --torch_dtype         "$TORCH_DTYPE" \
#     --attn_implementation sdpa \
#     --device_map          "$DEVICE_MAP"
# wait

# --- Adapter evaluation (OPSD checkpoint). ---
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python evaluate_aime_dllm.py \
    --base_model          "$BASE_MODEL" \
    --checkpoint_dir      "/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/dream7b-4epochs-30k/sft_dllm_dream7b_4epochs_30k/checkpoint-5600" \
    --dataset             "$DATASET" \
    --val_n               16 \
    --batch_size          2 \
    --max_new_tokens      768 \
    --diffusion_steps     768 \
    --temperature         0.5 \
    --top_p               0.95 \
    --alg                 entropy \
    --alg_temp            0.5 \
    --generator           diffusion \
    --torch_dtype         "$TORCH_DTYPE" \
    --attn_implementation sdpa \
    --device_map          "$DEVICE_MAP"
wait
