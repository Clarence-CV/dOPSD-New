#!/bin/bash

BASE_MODEL="Qwen/Qwen3-1.7B"

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4 
wait

# after trained, evaluate the performance of the trained model
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4 \
    --checkpoint_dir /home/stud_dat/on_policy_self_distill_dLLM/outputs/opsd_baseline_1b/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100
wait
    
