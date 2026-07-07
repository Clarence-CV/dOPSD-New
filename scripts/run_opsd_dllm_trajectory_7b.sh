
set -euo pipefail

cd "$(dirname "$0")/.."

export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
# GPU_IDS drives visible devices and accelerate --gpu_ids; NUM_PROCESSES = #ids.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra _GPU_ARR <<< "$GPU_IDS"
NUM_PROCESSES="${NUM_PROCESSES:-${#_GPU_ARR[@]}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"

STUDENT_BACKEND="${STUDENT_BACKEND:-dream}"
if [[ "$STUDENT_BACKEND" == "llada" ]]; then
    DEFAULT_MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
    BACKEND_TAG="llada8b"

    DEFAULT_MASK_TOKEN_ID=126336
    DEFAULT_ATTN_IMPL=eager
    DEFAULT_GRAD_CKPT=false
    DEFAULT_PER_DEVICE_BS=1
    DEFAULT_GRAD_ACCUM=4
elif [[ "$STUDENT_BACKEND" == "dream" ]]; then
    DEFAULT_MODEL_NAME="Dream-org/Dream-v0-Instruct-7B"
    BACKEND_TAG="dream7b"
    # -1 = use tokenizer default mask_token_id.
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

TRAJ_MASK_THRESHOLD="${TRAJ_MASK_THRESHOLD:-0.75}"
TRAJ_STEP_SELECT="${TRAJ_STEP_SELECT:-least}"
# Teacher view: -1 = concrete final rollout (endpoint); n>=0 = state n steps ahead.
TRAJ_TEACHER_GAP="${TRAJ_TEACHER_GAP:--1}"
# 'snapshot' (single forward, uses TRAJ_TEACHER_GAP) or 'all_future' (avg over
# future steps, one forward each -- expensive).
TRAJ_TEACHER_VIEW="${TRAJ_TEACHER_VIEW:-snapshot}"
if [[ "$TRAJ_TEACHER_VIEW" == "all_future" ]]; then
    VIEW_TAG="allfut"
elif [[ "$TRAJ_TEACHER_GAP" -lt 0 ]]; then
    VIEW_TAG="endpt"
else
    VIEW_TAG="gap${TRAJ_TEACHER_GAP}"
fi
TRAJ_TAG="traj${TRAJ_MASK_THRESHOLD//./}-${TRAJ_STEP_SELECT}-${VIEW_TAG}"

# USE_GRPO=true adds GRPO(group) term: L = JSD(correct) + grpo_coef*GRPO; false = pure JSD.
USE_GRPO="${USE_GRPO:-false}"
GRPO_NUM_ROLLOUTS="${GRPO_NUM_ROLLOUTS:-2}"
GRPO_COEF="${GRPO_COEF:-0.2}"
# true = verify-gated filter; wrong rollouts get no teacher signal (loss 0).
FILTER_WRONG_ROLLOUTS="${FILTER_WRONG_ROLLOUTS:-true}"
if [[ "$USE_GRPO" == "true" ]]; then
    MODE_TAG="GRPO"
else
    MODE_TAG="filterwrong"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/opsd_dllm_trajectory/}"

DATASET="mixchain"
DATA_TAG="mixchain"

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
