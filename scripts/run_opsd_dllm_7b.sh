#!/usr/bin/env bash
# OPSD self-distillation training for Dream-7B (a diffusion LM).
# Adapted from run_opsd_1b.sh:
#   * AR-only flags removed: --use_vllm / --vllm_* / --top_k / --lmbda / --max_completion_length.
#   * Dream uses its own diffusion_generate, controlled by --gen_max_new_tokens,
#     --gen_steps, --gen_temperature, --gen_top_p, --gen_alg, --gen_alg_temp.
#   * --max_length is dropped (Dream's prompt+answer length is set explicitly
#     by --max_prompt_length and --max_answer_length).
#   * --attn_implementation sdpa (Dream-7B is unsafe with flash_attention_2).
#
# MODE TOGGLE (env var):
#   OFF_POLICY=0 (default) — on-policy: the student rolls out a completion via
#                            diffusion_generate, which is then remasked.
#   OFF_POLICY=1           — off-policy: distill on the dataset's ground-truth
#                            answer (no rollout). Student sees the masked GT;
#                            teacher sees the concrete GT. The --gen_* flags are
#                            unused in this mode. Usage:
#                                OFF_POLICY=1 ./scripts/run_opsd_dllm_7b.sh
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# --- Mode toggle: on-policy rollout (default) vs off-policy GT distillation ---
# Separate run_config per mode so output_dir / W&B runs never collide.
OFF_POLICY="${OFF_POLICY:-0}"
if [[ "$OFF_POLICY" == "1" ]]; then
    OFF_POLICY_FLAG="--off_policy"
    RUN_CONFIG="dream7b_offpolicy_gt_v1"
    # Off-policy completion = GT answer; 1024 + 768 = 1792 keeps the forward
    # under Dream-v0's 2048 limit and matches the on-policy answer budget.
    MAX_ANSWER_LENGTH=1024
else
    OFF_POLICY_FLAG=""
    RUN_CONFIG="dream7b_gen768_forwardbeta0_v2"
    MAX_ANSWER_LENGTH=1024
fi
echo "[run_opsd_dllm_7b] OFF_POLICY=$OFF_POLICY  run_config=$RUN_CONFIG"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 1 \
    --main_process_port 13379 \
    opsd_dllm_train.py \
    --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
    --learning_rate 2e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --output_dir ./outputs/opsd_dllm/ \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 3 \
    --save_steps 50 \
    --logging_steps 2 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_answer_length "$MAX_ANSWER_LENGTH" \
    --gen_max_new_tokens 768 \
    --gen_steps 768 \
    --gen_temperature 0.5 \
    --gen_top_p 0.95 \
    --gen_alg entropy \
    --gen_alg_temp 0.5 \
    --beta 0 \
    --temperature 1.0 \
    --sampling_eps 1e-3 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj v_proj \
    --fixed_teacher \
    --jsd_token_clip 0.0 \
    $OFF_POLICY_FLAG \
    --wandb_project OPSD
