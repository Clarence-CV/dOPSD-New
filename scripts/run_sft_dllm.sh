#!/usr/bin/env bash
# Vanilla SFT for a diffusion LM (Dream-7B) via sft_train_dllm.py (masked-prediction loss).
# Toggles (env): MASK_SCHEDULE=diffusion|fixed (tune DIFF_MIN_T/DIFF_MAX_T or
# FIXED_MASK_RATIO="0.75"/"lo:hi").

set -euo pipefail

cd "$(dirname "$0")/.."

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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

MASK_SCHEDULE="${MASK_SCHEDULE:-fixed}"
FIXED_MASK_RATIO="${FIXED_MASK_RATIO:-0.5}"
DIFF_MIN_T="${DIFF_MIN_T:-0.0}"
DIFF_MAX_T="${DIFF_MAX_T:-1.0}"
case "$MASK_SCHEDULE" in
    diffusion)
        MASK_TAG="diff${DIFF_MIN_T//./}-${DIFF_MAX_T//./}"
        ;;
    fixed)
        MASK_TAG="fix${FIXED_MASK_RATIO//[:.]/}"
        ;;
    *)
        echo "[run_sft_dllm] ERROR: MASK_SCHEDULE must be 'diffusion' or 'fixed' (got '$MASK_SCHEDULE')" >&2
        exit 1
        ;;
esac

DATASET="mixchain"
DATA_TAG="mixchain12k"

RUN_CONFIG="sft_dllm_dream7b_${MASK_TAG}_2epochs_${DATA_TAG}"
OUTPUT_DIR="./outputs/sft_dllm/dream7b-${MASK_TAG}-2epochs-${DATA_TAG}"
echo "[run_sft_dllm] DATASET=$DATASET  MASK_SCHEDULE=$MASK_SCHEDULE  FIXED_MASK_RATIO=$FIXED_MASK_RATIO  run_config=$RUN_CONFIG"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 19347 \
    sft_train_dllm.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --dataset "$DATASET" \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 2 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_prompt_length 512 \
    --max_answer_length 256 \
    --remove_unused_columns false \
    --sampling_eps 1e-3 \
    --mask_schedule "$MASK_SCHEDULE" \
    --fixed_mask_ratio "$FIXED_MASK_RATIO" \
    --diffusion_min_t "$DIFF_MIN_T" \
    --diffusion_max_t "$DIFF_MAX_T" \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --logging_steps 10 \
    --save_steps 10000 \
    "${WANDB_ARGS[@]}"
