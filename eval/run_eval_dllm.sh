#!/bin/bash
# Multi-GPU AIME/MATH evaluation for Dream-7B (diffusion LM) OPSD checkpoints.
#
# Strategy: data-parallel sharding. Launches one Python process per GPU; each
# process loads the full model and evaluates a disjoint round-robin slice of
# the dataset. After all shards finish we merge JSONs and recompute aggregate
# metrics over the full problem set.
#
# Override GPUs:    GPU_LIST=0,1,2,3 ./run_eval_dllm.sh
# Override ckpt:    CHECKPOINT_DIR=/abs/path ./run_eval_dllm.sh
#
# HARD INVARIANTS (must match training — see scripts/run_opsd_dllm_7b.sh):
#   * --diffusion_steps MUST equal --max_new_tokens. steps < tokens forces the
#     diffusion sampler to commit multiple tokens per step → quality collapse.
#   * --max_new_tokens should match training's --gen_max_new_tokens (768) plus
#     stay under Dream-v0's 2048 position limit once the prompt is added.
#   * --generator diffusion is explicit: never silently fall back to AR generate.
set -euo pipefail

cd "$(dirname "$0")"          # cd into eval/ so `python evaluate_aime_dllm.py` resolves
REPO_ROOT="$(cd .. && pwd)"   # repo root, for absolute checkpoint paths

BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
DATASET="${DATASET:-math500}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Point this at a checkpoint from the new forward-KL run. Override with
# CHECKPOINT_DIR=/abs/path/... when invoking the script.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/stud_dat/on_policy_self_distill_dLLM/outputs/opsd_dllm/dream7b_gen768_forwardbeta0_v2/checkpoint-150}"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_SHARDS=${#GPUS[@]}

RUN_NAME="$(date +%Y%m%d_%H%M%S)_$(basename "$CHECKPOINT_DIR")_${DATASET}"
RESULTS_DIR="eval_results/${RUN_NAME}"
mkdir -p "$RESULTS_DIR"

echo "Launching $NUM_SHARDS shards across GPUs: ${GPU_LIST}"
echo "Run dir: $RESULTS_DIR"

PIDS=()
SHARD_FILES=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    OUT="$RESULTS_DIR/shard${i}of${NUM_SHARDS}.json"
    LOG="$RESULTS_DIR/shard${i}of${NUM_SHARDS}.log"
    SHARD_FILES+=("$OUT")

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPU python evaluate_aime_dllm.py \
        --base_model          "$BASE_MODEL" \
        --checkpoint_dir      "$CHECKPOINT_DIR" \
        --dataset             "$DATASET" \
        --val_n               16 \
        --batch_size          2 \
        --max_new_tokens      768 \
        --diffusion_steps     768 \
        --temperature         0.5 \
        --top_p               0.95 \
        --alg                 entropy \
        --alg_temp            0.5 \
        --generator           diffusion \
        --torch_dtype         "$TORCH_DTYPE" \
        --attn_implementation sdpa \
        --num_shards          "$NUM_SHARDS" \
        --shard_id            "$i" \
        --output_file         "$OUT" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  shard $i -> GPU $GPU (PID ${PIDS[$i]}, log $LOG)"
done

echo "Waiting for shards..."
FAILED=0
for idx in "${!PIDS[@]}"; do
    pid=${PIDS[$idx]}
    if wait "$pid"; then
        echo "  shard $idx OK"
    else
        echo "  shard $idx FAILED (see $RESULTS_DIR/shard${idx}of${NUM_SHARDS}.log)"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "One or more shards failed; not merging."
    exit 1
fi

echo "All shards complete. Merging..."
python merge_dllm_shards.py "${SHARD_FILES[@]}" \
    -o "$RESULTS_DIR/merged.json"
