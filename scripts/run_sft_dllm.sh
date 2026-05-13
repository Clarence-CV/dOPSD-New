#!/usr/bin/env bash
# Vanilla SFT for a diffusion LM (Dream-7B).
# Adapted from run_sft.sh:
#   * Entry script: sft_train_dllm.py (LLaDA-style masked-prediction loss).
#   * AR-only --max_length dropped; Dream's prompt+answer length is set
#     explicitly by --max_prompt_length and --max_answer_length.
#   * --attn_implementation sdpa (Dream-7B is unsafe with flash_attention_2).
#   * --torch_dtype bfloat16 + --sampling_eps added (diffusion ELBO knob).
#
# accelerate.yaml note: this repo's YAML pins `gradient_accumulation_steps: 1`.
# To bump GA, edit both the YAML and the --gradient_accumulation_steps flag below.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into OPSD/

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3,4}
export WANDB_PROJECT=${WANDB_PROJECT:-sft-dllm}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-online}

WANDB_ARGS=(--wandb_project "$WANDB_PROJECT")
if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ "$WANDB_MODE" == "disabled" ]]; then
    WANDB_ARGS+=(--disable_wandb)
fi

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 19347 \
    sft_train_dllm.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --output_dir ./outputs/sft_dllm/dream7b-4epochs-30k \
    --run_config sft_dllm_dream7b_4epochs_30k \
    --num_train_epochs 4 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_answer_length 1024 \
    --remove_unused_columns false \
    --sampling_eps 1e-3 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --logging_steps 5 \
    --save_steps 20 \
    "${WANDB_ARGS[@]}"
