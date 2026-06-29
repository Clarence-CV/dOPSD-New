#!/usr/bin/env bash
# Trajectory-OPSD self-distillation for a diffusion LM (Dream-7B or LLaDA-8B).
#
# Variant of run_opsd_dllm_7b.sh that drives opsd_dllm_trajectory_train.py:
#   * The student's noisy view is a REAL intermediate decoding step (the
#     least-masked step whose masked fraction still exceeds the threshold).
#   * The teacher's privileged information is the CONCRETE final rollout.
#   * On-policy only — there is no OFF_POLICY toggle and no synthetic
#     MASK_SCHEDULE here. Instead:
#       TRAJ_MASK_THRESHOLD=0.5 (default) — ">50% masked" step-eligibility cutoff.
#       TRAJ_STEP_SELECT=least  (default) — 'least' (closest to threshold),
#                                           'most' (noisiest), or 'random'.
#       TRAJ_TEACHER_GAP=-1     (default) — teacher view. -1 = concrete final
#                                           rollout (endpoint; teacher sees the
#                                           answer at every scored position). n>=0
#                                           = trajectory state n steps after the
#                                           student's step (honest peek-ahead).
#
#   STUDENT_BACKEND=dream (default) | llada
#   DATASET=mixchain (default) | zigeng
#
#   Usage:
#       ./scripts/run_opsd_dllm_trajectory_7b.sh
#       TRAJ_STEP_SELECT=most ./scripts/run_opsd_dllm_trajectory_7b.sh
#       STUDENT_BACKEND=llada ./scripts/run_opsd_dllm_trajectory_7b.sh
#
# HARD INVARIANTS (same as run_opsd_dllm_7b.sh):
#   * --beta 0 → forward KL (reverse KL collapses on-policy training).
#   * --gen_steps MUST equal --gen_max_new_tokens. A finer trajectory (more
#     steps) also gives more decoding states near the threshold to pick from.
#   * --max_prompt_length + --gen_max_new_tokens must stay under Dream-v0's
#     2048 position limit.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into OPSD/

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
# GPU selection. GPU_IDS drives both the visible devices and accelerate's
# --gpu_ids; NUM_PROCESSES defaults to the number of ids (one process per GPU).
#   8x A5000:  GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/run_opsd_dllm_trajectory_7b.sh
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra _GPU_ARR <<< "$GPU_IDS"
NUM_PROCESSES="${NUM_PROCESSES:-${#_GPU_ARR[@]}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"

# --- Backend toggle: Dream (default) vs LLaDA ---------------------------------
STUDENT_BACKEND="${STUDENT_BACKEND:-dream}"
if [[ "$STUDENT_BACKEND" == "llada" ]]; then
    DEFAULT_MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
    BACKEND_TAG="llada8b"
    # LLaDA's tokenizer does NOT expose mask_token_id; its [MASK] id is 126336.
    DEFAULT_MASK_TOKEN_ID=126336
    # LLaDA's modeling code has no SDPA path — must use eager attention.
    DEFAULT_ATTN_IMPL=eager
    # LLaDAModelLM does not implement gradient checkpointing.
    DEFAULT_GRAD_CKPT=false
    # No grad-ckpt + eager attention => high activation memory. Keep per-device
    # batch small and recover the effective batch via accumulation (1*4*4procs=16).
    DEFAULT_PER_DEVICE_BS=1
    DEFAULT_GRAD_ACCUM=4
elif [[ "$STUDENT_BACKEND" == "dream" ]]; then
    DEFAULT_MODEL_NAME="Dream-org/Dream-v0-Instruct-7B"
    BACKEND_TAG="dream7b"
    # Dream's tokenizer exposes mask_token_id; -1 = use tokenizer default.
    DEFAULT_MASK_TOKEN_ID=-1
    DEFAULT_ATTN_IMPL=sdpa
    DEFAULT_GRAD_CKPT=true
    DEFAULT_PER_DEVICE_BS=4
    DEFAULT_GRAD_ACCUM=1
else
    echo "[run_opsd_dllm_trajectory_7b] ERROR: STUDENT_BACKEND must be 'dream' or 'llada' (got '$STUDENT_BACKEND')" >&2
    exit 1
fi
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
MASK_TOKEN_ID="${MASK_TOKEN_ID:-$DEFAULT_MASK_TOKEN_ID}"
ATTN_IMPL="${ATTN_IMPL:-$DEFAULT_ATTN_IMPL}"
GRAD_CKPT="${GRAD_CKPT:-$DEFAULT_GRAD_CKPT}"
PER_DEVICE_BS="${PER_DEVICE_BS:-$DEFAULT_PER_DEVICE_BS}"
GRAD_ACCUM="${GRAD_ACCUM:-$DEFAULT_GRAD_ACCUM}"

# --- Trajectory knobs ---------------------------------------------------------
TRAJ_MASK_THRESHOLD="${TRAJ_MASK_THRESHOLD:-0.75}"
TRAJ_STEP_SELECT="${TRAJ_STEP_SELECT:-least}"
# Teacher view: -1 = concrete final rollout (endpoint); n>=0 = trajectory state
# n steps after the student's step (history[k+n], clamped to the final state).
TRAJ_TEACHER_GAP="${TRAJ_TEACHER_GAP:--1}"
# Teacher target: 'snapshot' (single forward; uses TRAJ_TEACHER_GAP) or
# 'all_future' (average over steps k+1->final; one forward per step — EXPENSIVE).
TRAJ_TEACHER_VIEW="${TRAJ_TEACHER_VIEW:-snapshot}"
if [[ "$TRAJ_TEACHER_VIEW" == "all_future" ]]; then
    VIEW_TAG="allfut"
elif [[ "$TRAJ_TEACHER_GAP" -lt 0 ]]; then
    VIEW_TAG="endpt"
else
    VIEW_TAG="gap${TRAJ_TEACHER_GAP}"
fi
TRAJ_TAG="traj${TRAJ_MASK_THRESHOLD//./}-${TRAJ_STEP_SELECT}-${VIEW_TAG}"

# --- GRPO knobs ---------------------------------------------------------------
# use_grpo: add the GRPO term so the model also learns from WRONG rollouts —
#   L = JSD(correct) + grpo_coef * GRPO(group). false = pure JSD(correct).
# grpo_num_rollouts: group size G (on-policy rollouts per prompt).
# grpo_coef: alpha in L = JSD + alpha*GRPO (0 = JSD only).
#
# OLD VERSION ("wrong rollout -> loss 0"): set USE_GRPO=false. With GRPO off and
# FILTER_WRONG_ROLLOUTS=true the loss is pure JSD on CORRECT rollouts only —
# wrong rollouts contribute exactly zero (verify-gated filter). This is the
# proven single-rollout path; identical machinery to the pre-GRPO trainer.
USE_GRPO="${USE_GRPO:-false}"
GRPO_NUM_ROLLOUTS="${GRPO_NUM_ROLLOUTS:-2}"
GRPO_COEF="${GRPO_COEF:-0.2}"
# Verify-gated filter: true = wrong rollouts get NO teacher signal (loss 0 there).
FILTER_WRONG_ROLLOUTS="${FILTER_WRONG_ROLLOUTS:-true}"
# Mode tag so old-version (filter-only) runs are not mislabeled as GRPO runs.
if [[ "$USE_GRPO" == "true" ]]; then
    MODE_TAG="GRPO"
else
    MODE_TAG="filterwrong"
fi

# Where checkpoints are written: <OUTPUT_DIR>/<RUN_CONFIG>/checkpoint-<step>/
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/opsd_dllm_trajectory/}"

# --- Dataset toggle: zigeng (default) vs mixchain -----------------------------
# zigeng: rollouts are verified against the dataset's "gt_answer" column
#         (DATASET_REGISTRY target_field in opsd_dllm_trajectory_train.py).
DATASET="${DATASET:-zigeng}"
case "$DATASET" in
    zigeng)   DATA_TAG="zigeng" ;;
    mixchain) DATA_TAG="mixchain" ;;
    *)
        echo "[run_opsd_dllm_trajectory_7b] ERROR: DATASET must be 'zigeng' or 'mixchain' (got '$DATASET')" >&2
        exit 1
        ;;
esac

RUN_CONFIG="${BACKEND_TAG}_${TRAJ_TAG}_${DATA_TAG}_reverseKL_${MODE_TAG}_v2"
echo "[run_opsd_dllm_trajectory_7b] STUDENT_BACKEND=$STUDENT_BACKEND  MODEL_NAME=$MODEL_NAME  DATASET=$DATASET  TRAJ_MASK_THRESHOLD=$TRAJ_MASK_THRESHOLD  TRAJ_STEP_SELECT=$TRAJ_STEP_SELECT  run_config=$RUN_CONFIG"
echo "[run_opsd_dllm_trajectory_7b] attn=$ATTN_IMPL  grad_ckpt=$GRAD_CKPT  per_device_bs=$PER_DEVICE_BS  grad_accum=$GRAD_ACCUM  mask_token_id=$MASK_TOKEN_ID"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROCESSES" \
    --gpu_ids "$GPU_IDS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --main_process_port 13379 \
    opsd_dllm_trajectory_train.py \
    --model_name_or_path "$MODEL_NAME" \
    --student_backend "$STUDENT_BACKEND" \
    --mask_token_id "$MASK_TOKEN_ID" \
    --dataset "$DATASET" \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --gradient_checkpointing "$GRAD_CKPT" \
    --output_dir ./outputs/opsd_dllm_trajectory/ \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 5 \
    --save_steps 100 \
    --logging_steps 2 \
    --attn_implementation "$ATTN_IMPL" \
    --torch_dtype bfloat16 \
    --max_prompt_length 512 \
    --max_answer_length 256 \
    --gen_max_new_tokens 256 \
    --gen_steps 256 \
    --gen_temperature 1.0 \
    --gen_top_p 0.95 \
    --gen_alg entropy \
    --gen_alg_temp 0.5 \
    --beta 1 \
    --temperature 1.0 \
    --sampling_eps 1e-3 \
    --traj_mask_threshold "$TRAJ_MASK_THRESHOLD" \
    --traj_step_select "$TRAJ_STEP_SELECT" \
    --traj_teacher_view "$TRAJ_TEACHER_VIEW" \
    --traj_teacher_gap "$TRAJ_TEACHER_GAP" \
    --use_grpo "$USE_GRPO" \
    --grpo_num_rollouts "$GRPO_NUM_ROLLOUTS" \
    --grpo_coef "$GRPO_COEF" \
    --filter_wrong_rollouts "$FILTER_WRONG_ROLLOUTS" \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --fixed_teacher \
    --jsd_token_clip 0.0 \
    --wandb_project OPSD
