#!/bin/bash
# Staged smoke test, run INSIDE an idev session on gpu-a100-dev (3x A100-40GB), from the repo root.
#   bash tacc/dev_smoke.sh env           # GPUs, imports, trl patch, bnb 4-bit, offline cache
#   bash tacc/dev_smoke.sh eval_trace    # base LLaDA, 8 GSM8K test problems, 1 GPU, with traces
#   bash tacc/dev_smoke.sh train         # 3-GPU d-OPSD on GSM8K for 2*BATCH_DIVIDE steps (ckpt every BATCH_DIVIDE)
#   bash tacc/dev_smoke.sh resume        # continue the same run to 3*BATCH_DIVIDE steps
#   bash tacc/dev_smoke.sh eval_adapter  # eval_trace with the latest LoRA checkpoint
#   bash tacc/dev_smoke.sh all
# Knob: BATCH_DIVIDE (default 8, the README's A100/H100 setting).
set -euo pipefail
cd "$(dirname "$0")/.."
source tacc/env.sh

DEV="$SCRATCH/dopsd/dev"
mkdir -p "$DEV"
BD="${BATCH_DIVIDE:-8}"

stage_env() {
    hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    python - <<'PY'
import os, torch, trl, bitsandbytes as bnb
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
src = open(os.path.join(os.path.dirname(trl.__file__), "trainer", "grpo_trainer.py")).read()
assert "range(1, global_batch_size + 1)" in src, "trl not patched: run tacc/setup_env.sh"
print("trl patched ok")
lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
print("bnb 4-bit forward ok", tuple(lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)).shape))
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
print("offline tokenizer ok; eos", t.eos_token_id)
PY
}

eval_run() {  # $1 = trace dir, rest = extra eval.py args
    local out="$1"; shift
    (cd eval && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 25011 eval.py \
        --dataset gsm --subsample 8 --batch_size 8 --gen_length 256 --diffusion_steps 128 \
        --block_length 32 --temperature 0.0 --output_dir "$DEV/generations" --trace_dir "$out" "$@")
    python analysis/summarize_traces.py "$out"
    du -sh "$out"
}

train_run() {
    { time NUM_GPUS=3 BATCH_DIVIDE=$BD SAVE_STEPS=$BD SAVE_TOTAL_LIMIT=1 ADAPTER_SAVE_STEPS=$BD \
        ACT_CKPT="${ACT_CKPT:-whole_layer}" TRACE_EVERY=1 RUN_NAME=dev OUTPUT_DIR="$DEV/train" \
        bash d-opsd/run/gsm/opsd.sh ; } 2>&1 | tee "$DEV/train_$1.log"
}

# Evaluate the newest bf16 adapter snapshot (the evaluation tier of the two-tier saving).
latest_ckpt() { ls -d "$DEV"/train/adapters/step-* | sort -t- -k2 -n | tail -1; }
show_saves() { du -sh "$DEV"/train/checkpoint-* "$DEV"/train/adapters/step-* 2>/dev/null; ls "$DEV/train/traces" | head; }

case "${1:-all}" in
    env) stage_env ;;
    eval_trace) eval_run "$DEV/trace_base" ;;
    train) rm -rf "$DEV/train"; MAX_STEPS=$((2 * BD)) train_run first; show_saves ;;
    resume) MAX_STEPS=$((3 * BD)) RESUME_FROM=auto train_run resume; show_saves ;;
    eval_adapter) eval_run "$DEV/trace_adapter" --checkpoint_path "$(latest_ckpt)" ;;
    all) stage_env; eval_run "$DEV/trace_base"
         rm -rf "$DEV/train"; MAX_STEPS=$((2 * BD)) train_run first
         MAX_STEPS=$((3 * BD)) RESUME_FROM=auto train_run resume
         eval_run "$DEV/trace_adapter" --checkpoint_path "$(latest_ckpt)" ;;
    *) echo "unknown stage $1"; exit 1 ;;
esac
