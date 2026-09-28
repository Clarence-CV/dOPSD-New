#!/bin/bash
# Paper-protocol eval (gen 256, 128 steps, block 32, greedy) + per-step traces, 3 GPUs via torchrun.
#   TASK=gsm sbatch tacc/eval.slurm                                   # base LLaDA
#   TASK=gsm CKPT=$SCRATCH/dopsd/runs/gsm_opsd/checkpoint-800 sbatch tacc/eval.slurm
# Optional: SUBSAMPLE=N (seeded subset), TRACE=0 (no traces).
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
TAG="${TAG:-$( [[ -n "$CKPT" ]] && echo "$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")" || echo base )}"
OUT="$SCRATCH/dopsd/eval/${TASK}_${TAG}"
EXTRA=()
[[ -n "$CKPT" ]] && EXTRA+=(--checkpoint_path "$CKPT")
[[ "${TRACE:-1}" == "1" ]] && EXTRA+=(--trace_dir "$OUT/traces")
[[ -n "${SUBSAMPLE:-}" ]] && EXTRA+=(--subsample "$SUBSAMPLE")
echo "[slurm] job=$SLURM_JOB_ID task=$TASK ckpt=${CKPT:-none} -> $OUT"

(cd eval && torchrun --nproc_per_node 3 --master_port 25005 eval.py \
    --dataset "$TASK" --batch_size 8 --gen_length 256 --diffusion_steps 128 --block_length 32 \
    --temperature 0.0 --remasking low_confidence --seed 42 --num_answer_per_question 1 --split test \
    --output_dir "$OUT/generations" ${EXTRA[@]+"${EXTRA[@]}"})

GEN_DIR=$(dirname "$(find "$OUT/generations" -name '*_generations.json' | head -1)")
(cd eval && python parse_and_get_acc.py "$GEN_DIR" "$TASK")
[[ "${TRACE:-1}" == "1" ]] && python analysis/summarize_traces.py "$OUT/traces"
du -sh "$OUT"
