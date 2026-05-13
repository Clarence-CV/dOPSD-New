#!/bin/bash

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
DATASET="${DATASET:-aime24}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Base dLLM evaluation on AIME.
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
    --base_model "$BASE_MODEL" \
    --dataset "$DATASET" \
    --val_n 1 \
    --batch_size 1 \
    --max_new_tokens 256 \
    --diffusion_steps 256 \
    --temperature 0.2 \
    --top_p 0.95 \
    --alg entropy \
    --alg_temp 0.0 \
    --torch_dtype "$TORCH_DTYPE" \
    --attn_implementation sdpa \
    --device_map "$DEVICE_MAP"
wait

# Adapter evaluation. Set CHECKPOINT_DIR=/path/to/checkpoint to enable.
if [ -n "$CHECKPOINT_DIR" ]; then
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python evaluate_aime_dllm.py \
        --base_model "$BASE_MODEL" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --dataset "$DATASET" \
        --val_n 1 \
        --batch_size 1 \
        --max_new_tokens 256 \
        --diffusion_steps 256 \
        --temperature 0.2 \
        --top_p 0.95 \
        --alg entropy \
        --alg_temp 0.0 \
        --torch_dtype "$TORCH_DTYPE" \
        --attn_implementation sdpa \
        --device_map "$DEVICE_MAP"
    wait
fi
