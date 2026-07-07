
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/tuan/local_home/ptuandat/on_policy_self_distill_dLLM/outputs/opsd_dllm_trajectory/eval_checkpoint/LLaDa/llada8b_traj05-least-allfut_mixchain_forwardbeta0_filterwrong_v2_1500}"
GPUS="${GPUS:-6}"
PORT="${PORT:-29600}"
TAG="$(basename "$MODEL_PATH")"
OUT_ROOT="${OUT_ROOT:-evals_results/$TAG}"

TASKS="${TASKS:-gsm8k_cot}"

echo "[eval/LLaDA] MODEL_PATH=$MODEL_PATH  GPUS=$GPUS  OUT_ROOT=$OUT_ROOT  TASKS='$TASKS'"

for TASK in $TASKS; do
  # Per-task generation params (mirrors the requested config).
  TOP_P=0.9
  TEMPERATURE=0.
  ALG=llada_original
  INCLUDE_PATH=""   # set for custom (non-installed) lm_eval tasks
  case "$TASK" in
    gsm8k_cot)
      LLADA_TASK=gsm8k
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
      DIFF_STEPS="${DIFF_STEPS:-256}"
      BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
      LOGITS_EOS_INF=False
      CONFIDENCE_EOS_EOT_INF=False
      NUM_FEWSHOT=0
      ;;
    humaneval_instruct)
      LLADA_TASK=humaneval
      MAX_NEW_TOKENS=512
      DIFF_STEPS=512
      BLOCK_LENGTH=512
      LOGITS_EOS_INF=True
      CONFIDENCE_EOS_EOT_INF=False
      NUM_FEWSHOT=0
      ;;
    minerva_math500)
      # MATH-500 subset — custom task registered via --include_path.
      LLADA_TASK=minerva_math500
      INCLUDE_PATH="${SCRIPT_DIR}/tasks/minerva_math500"
      MAX_NEW_TOKENS=512
      DIFF_STEPS=512
      BLOCK_LENGTH=64
      LOGITS_EOS_INF=False
      CONFIDENCE_EOS_EOT_INF=False
      NUM_FEWSHOT=0
      ;;
    minerva_math)
      # Full MATH (installed lm_eval task).
      LLADA_TASK=minerva_math
      MAX_NEW_TOKENS=512
      DIFF_STEPS=512
      BLOCK_LENGTH=64
      LOGITS_EOS_INF=False
      CONFIDENCE_EOS_EOT_INF=False
      NUM_FEWSHOT=0
      ;;
    mbpp_instruct)
      LLADA_TASK=mbpp
      MAX_NEW_TOKENS=256
      DIFF_STEPS=256
      BLOCK_LENGTH=256
      LOGITS_EOS_INF=False
      CONFIDENCE_EOS_EOT_INF=True
      NUM_FEWSHOT=0
      ;;
    *)
      echo "[eval/LLaDA] ERROR: unknown task '$TASK' (expected gsm8k_cot|humaneval_instruct|minerva_math500|mbpp_instruct)" >&2
      exit 1
      ;;
  esac

  OUT_DIR="${OUT_ROOT}/${TASK}-ns${NUM_FEWSHOT}-${MAX_NEW_TOKENS}"
  echo "[eval/LLaDA] task=$TASK -> llada_task=$LLADA_TASK  len=$MAX_NEW_TOKENS steps=$DIFF_STEPS block=$BLOCK_LENGTH logits_eos_inf=$LOGITS_EOS_INF confidence_eos_eot_inf=$CONFIDENCE_EOS_EOT_INF fewshot=$NUM_FEWSHOT"

  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch --main_process_port ${PORT} eval_llada.py \
    --tasks ${LLADA_TASK} --num_fewshot ${NUM_FEWSHOT} \
    ${INCLUDE_PATH:+--include_path "${INCLUDE_PATH}"} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path="${MODEL_PATH}",gen_length=${MAX_NEW_TOKENS},steps=${DIFF_STEPS},block_length=${BLOCK_LENGTH},logits_eos_inf=${LOGITS_EOS_INF},confidence_eos_eot_inf=${CONFIDENCE_EOS_EOT_INF},show_speed=True,task="${LLADA_TASK}" \
    --output_path ${OUT_DIR} --log_samples
done


