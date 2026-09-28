#!/bin/bash
cd d-opsd
export WANDB_PROJECT="opsd_gsm8k"
export LOGDIR="checkpoints"

mkdir -p $LOGDIR

DATASET="gsm8k"
RUN_NAME="${RUN_NAME:-opsd}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}" 
PASSK=8 
PASSK_TEMP=0.9 
TEACHER_RETAIN_RATIO=0.25
BATCH_DIVIDE="${BATCH_DIVIDE:-4}" # for A100 / H100, set to 8
# num_iter=barch_divide
TOP_K_LOSS=20
BETA=1
# debug1=True
fixed_teacher=True
add_ref=False
diff_student_mask=false
JSD_TOKEN_CLIP=0.05
# Cluster knobs (defaults reproduce the original script).
NUM_GPUS="${NUM_GPUS:-4}"               # accelerate.yaml num_processes
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/$DATASET/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:--1}"            # >0 overrides num_train_epochs
SAVE_STEPS="${SAVE_STEPS:-25}"          # opsd.yaml default; keep divisible by BATCH_DIVIDE
RESUME_FROM="${RESUME_FROM:-false}"     # checkpoint dir | auto | false
TRACE_EVERY="${TRACE_EVERY:-0}"         # dump a rollout trace every N generation rounds; 0 = off
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-500}"   # opsd.yaml default; 1 = keep only the latest full checkpoint
ADAPTER_SAVE_STEPS="${ADAPTER_SAVE_STEPS:-0}" # bf16 adapter-only snapshots every N steps; 0 = off
ACT_CKPT="${ACT_CKPT:-}"                # LLaDA activation checkpointing (e.g. whole_layer); '' = off
MASTER_PORT="${MASTER_PORT:-12356}"

if [ "$debug1" = "True" ]; then
    DEBUG_FLAG="--debug1"
else
    DEBUG_FLAG=""
fi
if [ "$fixed_teacher" = "True" ]; then
    FIXED_TEACHER_FLAG="--fixed_teacher"
else
    FIXED_TEACHER_FLAG=""
fi
if [ "$add_ref" = "True" ]; then
    ADD_REF_FLAG="--add_ref"
else
    ADD_REF_FLAG=""
fi
if [ "$diff_student_mask" = "True" ]; then
    DIFF_STUDENT_MASK_FLAG="--diff_student_mask"
else
    DIFF_STUDENT_MASK_FLAG=""
fi


accelerate launch \
    --config_file accelerate.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT d_opsd_train.py \
    --config opsd.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $BATCH_DIVIDE \
    --batch_divide $BATCH_DIVIDE \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --resume_from_checkpoint "$RESUME_FROM" \
    --trace_every $TRACE_EVERY \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --adapter_save_steps $ADAPTER_SAVE_STEPS \
    --activation_checkpointing "$ACT_CKPT" \
    --passk $PASSK \
    --passk_temperature $PASSK_TEMP \
    --teacher_retain_ratio $TEACHER_RETAIN_RATIO \
    --top_k_loss $TOP_K_LOSS \
    --beta $BETA \
    --jsd_token_clip $JSD_TOKEN_CLIP \
    $DEBUG_FLAG \
    $DIFF_STUDENT_MASK_FLAG \
    $ADD_REF_FLAG \
    $FIXED_TEACHER_FLAG \
    "$@"
