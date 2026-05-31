#!/usr/bin/env bash
# OPSD self-distillation training for a diffusion LM (Dream-7B or LLaDA-8B).
# Adapted from run_opsd_1b.sh:
#   * AR-only flags removed: --use_vllm / --vllm_* / --top_k / --lmbda / --max_completion_length.
#   * Dream uses its own diffusion_generate; LLaDA uses the trainer's built-in
#     progressive-unmasking sampler. Both are controlled by --gen_max_new_tokens,
#     --gen_steps, --gen_temperature, --gen_top_p. (--gen_alg / --gen_alg_temp
#     are Dream-only knobs; they are ignored on the LLaDA path.)
#   * --max_length is dropped (prompt+answer length is set explicitly by
#     --max_prompt_length and --max_answer_length).
#   * --attn_implementation sdpa (Dream-7B is unsafe with flash_attention_2).
#
# MODE TOGGLES (env vars):
#   OFF_POLICY=0 (default) — on-policy: the student rolls out a completion,
#                            which is then remasked.
#   OFF_POLICY=1           — off-policy: distill on the dataset's ground-truth
#                            answer (no rollout). The --gen_* flags are unused.
#
#   STUDENT_BACKEND=dream  (default) — Dream-org/Dream-v0-Instruct-7B (AutoModel).
#   STUDENT_BACKEND=llada            — GSAI-ML/LLaDA-8B-Instruct (AutoModelForCausalLM).
#                                      Overrides the model path; pass MODEL_NAME=...
#                                      to point at a different LLaDA checkpoint.
#
#   MASK_SCHEDULE=diffusion (default) — antithetic per-example rate, i.i.d.
#                                       Bernoulli per valid position. Tune with
#                                       DIFF_MIN_T / DIFF_MAX_T (default 0/1).
#   MASK_SCHEDULE=fixed               — exact-count k = round(n_valid * ratio).
#                                       Tune with FIXED_MASK_RATIO ("0.75" or
#                                       a "lo:hi" range like "0.25:0.75").
#
#   DATASET=zigeng   (default) — Zigeng/dParallel_Dream_Distill_Data.
#   DATASET=mixchain           — horseee/MixChain-Z-PRM12K (question -> problem,
#                                answer -> solution).
#
#   Usage examples:
#       ./scripts/run_opsd_dllm_7b.sh
#       DATASET=mixchain ./scripts/run_opsd_dllm_7b.sh
#       OFF_POLICY=1 ./scripts/run_opsd_dllm_7b.sh
#       STUDENT_BACKEND=llada ./scripts/run_opsd_dllm_7b.sh
#       STUDENT_BACKEND=llada OFF_POLICY=1 ./scripts/run_opsd_dllm_7b.sh
#       MASK_SCHEDULE=fixed FIXED_MASK_RATIO=0.5 ./scripts/run_opsd_dllm_7b.sh
#
# HARD INVARIANTS for Dream OPSD (do not break these):
#   * --beta 0  → forward KL. Reverse KL (--beta 1) is zero-forcing and causes
#     on-policy mode collapse (student degenerates to repeated tokens).
#   * --gen_steps MUST equal --gen_max_new_tokens. steps < tokens forces the
#     diffusion sampler to commit multiple tokens per step → quality collapse.
#     (On-policy only; the --gen_* flags are ignored when OFF_POLICY=1.)
#   * The student/teacher forward length must stay within Dream-v0's 2048
#     position limit:
#       on-policy : --max_prompt_length + --gen_max_new_tokens  (1024 + 768 = 1792)
#       off-policy: --max_prompt_length + --max_answer_length   (1024 + 768 = 1792)
#   * LLaDA has a larger context window (8192 for LLaDA-8B-Instruct), so the
#     2048 limit doesn't bind there, but keeping the same budget makes the
#     two runs directly comparable.
#   * Eval (eval/run_eval_dllm.sh) must use the same generation length and the
#     same steps==tokens rule, otherwise train/eval mismatch.
#
# accelerate.yaml note: this repo's YAML has been updated to use a literal
# `gradient_accumulation_steps: 1` (the prior 'auto' placeholder failed to
# resolve in the installed accelerate version). If you want to bump GA, edit
# both the YAML and the --gradient_accumulation_steps flag below.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into OPSD/

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}

# --- Backend toggle: Dream (default) vs LLaDA ---------------------------------
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

# --- Mask-schedule toggle: diffusion (default) vs fixed -----------------------
MASK_SCHEDULE="${MASK_SCHEDULE:-fixed}"
FIXED_MASK_RATIO="${FIXED_MASK_RATIO:-0.5}"
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
        echo "[run_opsd_dllm_7b] ERROR: MASK_SCHEDULE must be 'diffusion' or 'fixed' (got '$MASK_SCHEDULE')" >&2
        exit 1
        ;;
esac

# --- Dataset toggle: zigeng (default) vs mixchain -----------------------------
# zigeng   = Zigeng/dParallel_Dream_Distill_Data (question -> problem, llm_response -> solution).
# mixchain = horseee/MixChain-Z-PRM12K          (question -> problem, answer       -> solution).
DATASET="${DATASET:-mixchain}"
case "$DATASET" in
    zigeng)   DATA_TAG="zigeng" ;;
    mixchain) DATA_TAG="mixchain" ;;
    *)
        echo "[run_opsd_dllm_7b] ERROR: DATASET must be 'zigeng' or 'mixchain' (got '$DATASET')" >&2
        exit 1
        ;;
esac

# --- Mode toggle: on-policy rollout (default) vs off-policy GT distillation ---
# Separate run_config per mode (and dataset) so output_dir / W&B runs never collide.
OFF_POLICY="${OFF_POLICY:-0}"
if [[ "$OFF_POLICY" == "1" ]]; then
    OFF_POLICY_FLAG="--off_policy"
    RUN_CONFIG="${BACKEND_TAG}_${MASK_TAG}_${DATA_TAG}_offpolicy_gt_v1"
    # Off-policy completion = GT answer; 1024 + 768 = 1792 keeps the forward
    # under Dream-v0's 2048 limit and matches the on-policy answer budget.
    MAX_ANSWER_LENGTH=1024
else
    OFF_POLICY_FLAG=""
    RUN_CONFIG="${BACKEND_TAG}_${MASK_TAG}_${DATA_TAG}_gen768_forwardbeta0_v2"
    MAX_ANSWER_LENGTH=1024
fi
echo "[run_opsd_dllm_7b] STUDENT_BACKEND=$STUDENT_BACKEND  MODEL_NAME=$MODEL_NAME  DATASET=$DATASET  MASK_SCHEDULE=$MASK_SCHEDULE  OFF_POLICY=$OFF_POLICY  run_config=$RUN_CONFIG"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gpu_ids "${CUDA_VISIBLE_DEVICES:-1,2,3,4}" \
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
    --save_steps 10 \
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
    --wandb_project OPSD
