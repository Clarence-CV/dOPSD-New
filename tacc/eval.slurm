#!/bin/bash
# Paper-protocol eval (gen 256, 128 steps, block 32, greedy) + per-step traces, 3 GPUs via torchrun.
#   TASK=gsm sbatch tacc/eval.slurm                                   # base LLaDA
#   TASK=gsm CKPT=$SCRATCH/dopsd/runs/gsm_opsd/checkpoint-800 sbatch tacc/eval.slurm
# Optional: SUBSAMPLE=N (seeded subset), TRACE=0 (no traces),
#   STEPS=64|32 (4|8 tokens revealed per step; paper/training use 128), TEMP=0.9 (training's sampling temp),
#   THRESHOLD=0.9 (confidence-threshold parallel decoding instead of the fixed budget; STEPS is then ignored).
#SBATCH --job-name=dopsd-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tacc/env.sh

TASK="${TASK:-gsm}"
CKPT="${CKPT:-}"
STEPS="${STEPS:-128}"
TEMP="${TEMP:-0.0}"
THRESHOLD="${THRESHOLD:-0}"
TAG="${TAG:-$( [[ -n "$CKPT" ]] && echo "$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT")" || echo base )_s${STEPS}_t${TEMP}$( [[ "$THRESHOLD" != "0" ]] && echo "_thr${THRESHOLD}" )}"
OUT="$SCRATCH/dopsd/eval/${TASK}_${TAG}"
EXTRA=()
[[ -n "$CKPT" ]] && EXTRA+=(--checkpoint_path "$CKPT")
[[ "${TRACE:-1}" == "1" ]] && EXTRA+=(--trace_dir "$OUT/traces")
[[ -n "${SUBSAMPLE:-}" ]] && EXTRA+=(--subsample "$SUBSAMPLE")
[[ "$THRESHOLD" != "0" ]] && EXTRA+=(--threshold "$THRESHOLD")
echo "[slurm] job=$SLURM_JOB_ID task=$TASK ckpt=${CKPT:-none} -> $OUT"

(cd eval && torchrun --nproc_per_node 3 --master_port 25005 eval.py \
    --dataset "$TASK" --batch_size 8 --gen_length 256 --diffusion_steps "$STEPS" --block_length 32 \
    --temperature "$TEMP" --remasking low_confidence --seed 42 --num_answer_per_question 1 --split test \
    --output_dir "$OUT/generations" ${EXTRA[@]+"${EXTRA[@]}"})

GEN_DIR=$(dirname "$(find "$OUT/generations" -name '*_generations.json' | head -1)")
(cd eval && python parse_and_get_acc.py "$GEN_DIR" "$TASK")
[[ "${TRACE:-1}" == "1" ]] && python analysis/summarize_traces.py "$OUT/traces"
du -sh "$OUT"
