#!/usr/bin/env bash
# Vanilla SFT for a diffusion LM (Dream-7B).
# Adapted from run_sft.sh:
#   * Entry script: sft_train_dllm.py (LLaDA-style masked-prediction loss).
#   * AR-only --max_length dropped; Dream's prompt+answer length is set
#     explicitly by --max_prompt_length and --max_answer_length.
#   * --attn_implementation sdpa (Dream-7B is unsafe with flash_attention_2).
#   * --torch_dtype bfloat16 + --sampling_eps added (diffusion ELBO knob).
#
# MASK-SCHEDULE TOGGLE (env vars), mirrors run_opsd_dllm_7b.sh:
#   MASK_SCHEDULE=diffusion (default) — antithetic per-example rate t ∈
#                                       [DIFF_MIN_T, DIFF_MAX_T], i.i.d.
#                                       Bernoulli per valid answer position.
#                                       This is the LLaDA ELBO default.
#   MASK_SCHEDULE=fixed               — exact-count k = round(n_answer * ratio)
#                                       positions per example. Tune with
#                                       FIXED_MASK_RATIO ("0.75" or "lo:hi"
#                                       like "0.25:0.75").
#
#   Usage examples:
#       ./scripts/run_sft_dllm.sh
#       MASK_SCHEDULE=fixed FIXED_MASK_RATIO=0.5 ./scripts/run_sft_dllm.sh
#       MASK_SCHEDULE=fixed FIXED_MASK_RATIO=0.25:0.75 ./scripts/run_sft_dllm.sh
#
# accelerate.yaml note: this repo's YAML pins `gradient_accumulation_steps: 1`.
# To bump GA, edit both the YAML and the --gradient_accumulation_steps flag below.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into OPSD/

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}
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

# --- Mask-schedule toggle: diffusion (default) vs fixed -----------------------
MASK_SCHEDULE="${MASK_SCHEDULE:-fixed}"
FIXED_MASK_RATIO="${FIXED_MASK_RATIO:-0.75}"
DIFF_MIN_T="${DIFF_MIN_T:-0.0}"
DIFF_MAX_T="${DIFF_MAX_T:-1.0}"
case "$MASK_SCHEDULE" in
    diffusion)
        MASK_TAG="diff${DIFF_MIN_T//./}-${DIFF_MAX_T//./}"
        ;;
    fixed)
        # Replace ':' / '.' so the tag is filesystem-safe.
        MASK_TAG="fix${FIXED_MASK_RATIO//[:.]/}"
        ;;
    *)
        echo "[run_sft_dllm] ERROR: MASK_SCHEDULE must be 'diffusion' or 'fixed' (got '$MASK_SCHEDULE')" >&2
        exit 1
        ;;
esac

RUN_CONFIG="sft_dllm_dream7b_${MASK_TAG}_2epochs_30k"
OUTPUT_DIR="./outputs/sft_dllm/dream7b-${MASK_TAG}-2epochs-30k"
echo "[run_sft_dllm] MASK_SCHEDULE=$MASK_SCHEDULE  FIXED_MASK_RATIO=$FIXED_MASK_RATIO  run_config=$RUN_CONFIG"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 19347 \
    sft_train_dllm.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
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
    --save_steps 100 \
    "${WANDB_ARGS[@]}"
