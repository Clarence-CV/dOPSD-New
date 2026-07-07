#!/usr/bin/env bash
# OPSD self-distillation for a diffusion LM (Dream-7B or LLaDA-8B).
# Toggles (env): OFF_POLICY, STUDENT_BACKEND=dream|llada, MASK_SCHEDULE=diffusion|fixed,
# USE_PI. See run_config assembly below for tags.
# Invariants: --beta 0 = forward KL (reverse KL collapses on-policy);
# --gen_steps MUST equal --gen_max_new_tokens; max_prompt_length + gen len must
# stay under Dream-v0's 2048 limit; eval must match the same gen len / steps rule.

set -euo pipefail

cd "$(dirname "$0")/.."

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}

STUDENT_BACKEND="${STUDENT_BACKEND:-dream}"
if [[ "$STUDENT_BACKEND" == "llada" ]]; then
    DEFAULT_MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
    BACKEND_TAG="llada8b"
elif [[ "$STUDENT_BACKEND" == "dream" ]]; then
    DEFAULT_MODEL_NAME="Dream-org/Dream-v0-Instruct-7B"
    BACKEND_TAG="dream7b"
else
    echo "[run_opsd_dllm_7b] ERROR: STUDENT_BACKEND must be 'dream' or 'llada' (got '$STUDENT_BACKEND')" >&2
    exit 1
fi
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"

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
        echo "[run_opsd_dllm_7b] ERROR: MASK_SCHEDULE must be 'diffusion' or 'fixed' (got '$MASK_SCHEDULE')" >&2
        exit 1
        ;;
esac

DATASET="mixchain"
DATA_TAG="mixchain"

OFF_POLICY="${OFF_POLICY:-0}"
if [[ "$OFF_POLICY" == "1" ]]; then
    OFF_POLICY_FLAG="--off_policy"
    POLICY_TAG="offpolicy"
    MAX_ANSWER_LENGTH=1024
else
    OFF_POLICY_FLAG=""
    POLICY_TAG="onpolicy"
    MAX_ANSWER_LENGTH=1024
fi

USE_PI="${USE_PI:-1}"
if [[ "$USE_PI" == "1" ]]; then
    PI_FLAG="--use_privileged_info"
    PI_TAG="PI"
else
    PI_FLAG=""
    PI_TAG="noPI"
fi

RUN_CONFIG="${BACKEND_TAG}_${MASK_TAG}_${DATA_TAG}_${POLICY_TAG}_${PI_TAG}_beta0_v2"
echo "[run_opsd_dllm_7b] STUDENT_BACKEND=$STUDENT_BACKEND  MODEL_NAME=$MODEL_NAME  DATASET=$DATASET  MASK_SCHEDULE=$MASK_SCHEDULE  OFF_POLICY=$OFF_POLICY  USE_PI=$USE_PI  run_config=$RUN_CONFIG"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gpu_ids 1,2,3,4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 13379 \
    opsd_dllm_train.py \
    --model_name_or_path "$MODEL_NAME" \
    --student_backend "$STUDENT_BACKEND" \
    --dataset "$DATASET" \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --output_dir ./outputs/opsd_dllm/ \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 3 \
    --save_steps 100 \
    --logging_steps 2 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_prompt_length 512 \
    --max_answer_length 256 \
    --gen_max_new_tokens 256 \
    --gen_steps 256 \
    --gen_temperature 1.0 \
    --gen_top_p 0.95 \
    --gen_alg entropy \
    --gen_alg_temp 0.5 \
    --beta 0 \
    --temperature 1.0 \
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
    --fixed_teacher \
    --jsd_token_clip 0.0 \
    $OFF_POLICY_FLAG \
    $PI_FLAG \
    --wandb_project OPSD
