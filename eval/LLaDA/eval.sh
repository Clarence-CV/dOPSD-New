# Evaluate a TRAINED LLaDA model on the same four benchmark families used for
# Dream (see eval/Dream/eval_instruct/eval.sh): gsm8k, minerva_math, humaneval,
# mbpp. LLaDA uses eval_llada.py with --model llada_dist; the model is selected
# via `model_path=`.
#
# IMPORTANT — merge LoRA first. eval_llada.py loads a FULL model
# (LLaDAModelLM.from_pretrained), so a raw LoRA adapter will NOT work. Merge the
# trained adapter into the base model with merge_lora.py, then point MODEL_PATH
# at the merged dir. Keep "instruct" in the merged dir name so the chat template
# is applied (eval_llada.py auto-detects is_instruct from the path).
#
# ALG note: alg=llada_original == the original LLaDA decoding (low_confidence
# remasking, no dParallel entropy threshold). This harness uses that by default
# (remasking='low_confidence' and NO threshold passed). temperature is 0 and
# top_p is unused by LLaDA's gumbel-argmax sampling, so they are informational.
#
#   Usage:
#       bash eval/LLaDA/eval.sh                              # all four tasks
#       TASKS="gsm8k_cot mbpp_instruct" bash eval/LLaDA/eval.sh
#       MODEL_PATH=/abs/path/to/merged GPUS=0,1,2,3 bash eval/LLaDA/eval.sh

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# --- Model under test ---------------------------------------------------------
# Merged checkpoint of the trained LLaDA run
# (run_config: llada8b_traj05-least-endpt_mixchain_forwardbeta0_filterwrong_v2).
# Must be a FULL merged model dir whose name contains "instruct".
MODEL_PATH="${MODEL_PATH:-/home/stud_dat/on_policy_self_distill_dLLM/outputs/opsd_dllm_trajectory/llada8b_traj05-least-endpt_mixchain_forwardbeta0_filterwrong_v2_800_merge}"
GPUS="${GPUS:-0}"
PORT="${PORT:-29600}"
# Short tag for output dirs (basename of the model path).
TAG="$(basename "$MODEL_PATH")"
OUT_ROOT="${OUT_ROOT:-evals_results/$TAG}"

# Tasks to run (use the Dream-style labels; mapped to LLaDA task names below).
TASKS="${TASKS:-gsm8k_cot minerva_math500 humaneval_instruct mbpp_instruct}"

echo "[eval/LLaDA] MODEL_PATH=$MODEL_PATH  GPUS=$GPUS  OUT_ROOT=$OUT_ROOT  TASKS='$TASKS'"

for TASK in $TASKS; do
  # Per-task generation params (mirrors the requested config).
  TOP_P=0.9
  TEMPERATURE=0.
  ALG=llada_original
  case "$TASK" in
    gsm8k_cot)
      LLADA_TASK=gsm8k
      # export BLOCK_LENGTH=32 to match trajectory blocks (default 8 for legacy reproducibility)
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
    minerva_math|minerva_math500)
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
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path="${MODEL_PATH}",gen_length=${MAX_NEW_TOKENS},steps=${DIFF_STEPS},block_length=${BLOCK_LENGTH},logits_eos_inf=${LOGITS_EOS_INF},confidence_eos_eot_inf=${CONFIDENCE_EOS_EOT_INF},show_speed=True,task="${LLADA_TASK}" \
    --output_path ${OUT_DIR} --log_samples
done

## NOTICE: the code tasks need postprocessing before scoring — pass the
## generated samples_*.jsonl under the matching output_path:
#   python postprocess_code_humaneval.py ${OUT_ROOT}/humaneval_instruct-ns0-512/.../samples_*.jsonl
#   python postprocess_code_mbpp.py       ${OUT_ROOT}/mbpp_instruct-ns0-256/.../samples_*.jsonl
