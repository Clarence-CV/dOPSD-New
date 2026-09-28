#!/bin/bash
# d-OPSD (LLaDA-8B-Instruct, 4-bit LoRA), 1 node x 3 A100-40GB (gpu-a100, 3 SU/node-hour).
# Submit from the repo root (logs/ must exist):
#   TASK=gsm MAX_STEPS=800 sbatch --mail-user=you@x --mail-type=BEGIN,END,FAIL tacc/train.slurm
# One optimizer step = one BATCH_DIVIDE-th of each GPU's trajectory; a new rollout
# per GPU every BATCH_DIVIDE steps. Keep MAX_STEPS / SAVE_STEPS multiples of BATCH_DIVIDE.
# Relay past the wall clock (same RUN_NAME, resumes from the latest checkpoint):
#   JOB1=$(sbatch --parsable tacc/train.slurm)
#   RESUME_FROM=auto sbatch --dependency=afterany:$JOB1 tacc/train.slurm
#SBATCH --job-name=dopsd-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tacc/env.sh

TASK="${TASK:-gsm}"
export RUN_NAME="${RUN_NAME:-${TASK}_opsd}"
export OUTPUT_DIR="$SCRATCH/dopsd/runs/$RUN_NAME"
export NUM_GPUS=3
export BATCH_DIVIDE="${BATCH_DIVIDE:-8}"
export MAX_STEPS="${MAX_STEPS:?set MAX_STEPS (multiple of BATCH_DIVIDE)}"
export SAVE_STEPS="${SAVE_STEPS:-128}"
export TRACE_EVERY="${TRACE_EVERY:-25}"
export ACT_CKPT="${ACT_CKPT:-whole_layer}"            # A100-40GB OOMs without it (paper used 80 GB cards)
# Two-tier saving: frequent bf16 adapter-only snapshots for eval; one rolling full checkpoint for resume.
export ADAPTER_SAVE_STEPS="${ADAPTER_SAVE_STEPS:-64}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export RESUME_FROM="${RESUME_FROM:-false}"
echo "[slurm] job=$SLURM_JOB_ID node=$(hostname) task=$TASK run=$RUN_NAME max_steps=$MAX_STEPS batch_divide=$BATCH_DIVIDE save=$SAVE_STEPS trace_every=$TRACE_EVERY resume=$RESUME_FROM"
nvidia-smi --query-gpu=name,memory.total --format=csv

bash "d-opsd/run/$TASK/opsd.sh"

# $SCRATCH is purged: keep adapters + traces (not DeepSpeed optimizer shards) on $WORK.
mkdir -p "$DOPSD_ROOT/runs"
rsync -a --exclude 'global_step*' "$OUTPUT_DIR/" "$DOPSD_ROOT/runs/$RUN_NAME/"
echo "[slurm] copied to $DOPSD_ROOT/runs/$RUN_NAME"
