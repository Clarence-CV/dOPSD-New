#!/bin/bash

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
DATASET="${DATASET:-aime24}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Base dLLM evaluation on AIME.
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
#     --base_model "$BASE_MODEL" \
#     --dataset "$DATASET" \
#     --val_n 12 \
#     --batch_size 2 \
#     --max_new_tokens 256 \
#     --diffusion_steps 256 \
#     --temperature 0.2 \
#     --top_p 0.95 \
#     --alg entropy \
#     --alg_temp 0.0 \
#     --torch_dtype "$TORCH_DTYPE" \
#     --attn_implementation sdpa \
#     --device_map "$DEVICE_MAP"
# wait

# Adapter evaluation. Set CHECKPOINT_DIR=/path/to/checkpoint to enable.
#if [ -n "$CHECKPOINT_DIR" ]; then
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir //home/stud_dat/on_policy_self_distill_dLLM/outputs/opsd_dllm/dream7b_gen256_fixteacher_forwardbeta0_clip005/checkpoint-100 \
    --dataset "$DATASET" \
    --val_n 12 \
    --batch_size 2 \
    --max_new_tokens 2048 \
    --diffusion_steps 512 \
    --temperature 0.2 \
    --top_p 0.95 \
    --alg entropy \
    --alg_temp 0.0 \
    --torch_dtype "$TORCH_DTYPE" \
    --attn_implementation sdpa \
    --device_map "$DEVICE_MAP"
wait

# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
#     --base_model "$BASE_MODEL" \
#     --checkpoint_dir /home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/dream7b-4epochs-30k/sft_dllm_dream7b_4epochs_30k/checkpoint-600 \
#     --dataset "$DATASET" \
#     --val_n 12 \
#     --batch_size 2 \
#     --max_new_tokens 256 \
#     --diffusion_steps 256 \
#     --temperature 0.2 \
#     --top_p 0.95 \
#     --alg entropy \
#     --alg_temp 0.0 \
#     --torch_dtype "$TORCH_DTYPE" \
#     --attn_implementation sdpa \
#     --device_map "$DEVICE_MAP"
# wait

# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
#     --base_model "$BASE_MODEL" \
#     --checkpoint_dir /home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/dream7b-4epochs-30k/sft_dllm_dream7b_4epochs_30k/checkpoint-3480 \
#     --dataset "$DATASET" \
#     --val_n 12 \
#     --batch_size 2 \
#     --max_new_tokens 256 \
#     --diffusion_steps 256 \
#     --temperature 0.2 \
#     --top_p 0.95 \
#     --alg entropy \
#     --alg_temp 0.0 \
#     --torch_dtype "$TORCH_DTYPE" \
#     --attn_implementation sdpa \
#     --device_map "$DEVICE_MAP"
# wait