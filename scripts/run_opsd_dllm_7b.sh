#!/usr/bin/env bash
# OPSD self-distillation training for Dream-7B (a diffusion LM).
# Adapted from run_opsd_1b.sh:
#   * AR-only flags removed: --use_vllm / --vllm_* / --top_k / --lmbda / --max_completion_length.
#   * Dream uses its own diffusion_generate, controlled by --gen_max_new_tokens,
#     --gen_steps, --gen_temperature, --gen_top_p, --gen_alg, --gen_alg_temp.
#   * --max_length is dropped (Dream's prompt+answer length is set explicitly
#     by --max_prompt_length and --max_answer_length).
#   * --attn_implementation sdpa (Dream-7B is unsafe with flash_attention_2).
#
# accelerate.yaml note: this repo's YAML has been updated to use a literal
# `gradient_accumulation_steps: 1` (the prior 'auto' placeholder failed to
# resolve in the installed accelerate version). If you want to bump GA, edit
# both the YAML and the --gradient_accumulation_steps flag below.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into OPSD/

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 13379 \
    opsd_dllm_train.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --output_dir ./outputs/opsd_dllm/ \
    --run_config dream7b_gen256_fixteacher_forwardbeta0_clip005 \
    --num_train_epochs 30 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_answer_length 1024 \
    --gen_max_new_tokens 512 \
    --gen_steps 512 \
    --gen_temperature 0.2 \
    --gen_top_p 0.95 \
    --gen_alg entropy \
    --gen_alg_temp 0.0 \
    --beta 1 \
    --temperature 1.0 \
    --sampling_eps 1e-3 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
