#!/usr/bin/env bash
# AR(Qwen3) -> dLLM(Dream-7B) on-policy distillation.
#
# Teacher = Qwen3 (frozen autoregressive LM); Student = Dream-7B (diffusion LM,
# trained with LoRA). Each step: the teacher analyses the reference solution
# (AR generate), the student rolls out a completion (diffusion_generate), the
# completion is randomly remasked, and JSD is taken at the masked positions
# between the student's denoising distribution and the teacher's AR next-token
# distribution.
#
# HARD INVARIANTS for the Dream student (do not break these):
#   * --beta 0  -> forward KL. Reverse KL (--beta 1) is zero-forcing and causes
#     on-policy mode collapse (student degenerates to repeated tokens).
#   * --gen_steps MUST equal --gen_max_new_tokens. steps < tokens forces the
#     diffusion sampler to commit multiple tokens per step -> quality collapse.
#   * --max_prompt_length + --gen_max_new_tokens MUST stay < 2048 (Dream-v0's
#     position-embedding limit). Here: 1024 + 768 = 1792.
#   * --attn_implementation sdpa for the student (Dream-7B is unsafe with FA2).
#
# Memory note: this trainer holds TWO ~7-8B models per GPU (student + frozen
# teacher). per_device_train_batch_size is kept at 1; raise only if VRAM allows.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into the repo root

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 13380 \
    opsd_ar2dllm_train.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --teacher_model_name_or_path Qwen/Qwen3-8B \
    --teacher_attn_implementation sdpa \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --output_dir ./outputs/opsd_ar2dllm/ \
    --run_config qwen3_8b_to_dream7b_gen768_forwardbeta0 \
    --num_train_epochs 3 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_teacher_prompt_length 3072 \
    --max_reasoning_length 2048 \
    --reason_first True \
    --teacher_gen_temperature 0.7 \
    --teacher_gen_top_p 0.95 \
    --teacher_gen_top_k 20 \
    --gen_max_new_tokens 768 \
    --gen_steps 768 \
    --gen_temperature 0.5 \
    --gen_top_p 0.95 \
    --gen_alg entropy \
    --gen_alg_temp 0.5 \
    --beta 0 \
    --temperature 1.0 \
    --sampling_eps 1e-3 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
