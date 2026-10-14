#!/bin/bash
# Batch entry for one Selective Draft stage. Everything is an environment-variable knob; nothing defaults to
# a long run. Budget before submitting: SU = nodes x max(elapsed h, 0.25) x rate (gpu-a100 3, gpu-a100-small 1.5).
#   STAGE=decode N_COLLECT=200 sbatch -p gpu-a100-small -t 02:00:00 tacc/selective_draft.slurm
#   STAGE=readout MAX_STEPS=400 sbatch -p gpu-a100-small -t 03:00:00 tacc/selective_draft.slurm
#   STAGE=guidance MAX_STEPS=100 sbatch -p gpu-a100 -t 04:00:00 tacc/selective_draft.slurm
# Any STAGE accepted by tacc/selective_draft_smoke.sh works; OUT defaults to $SCRATCH/dopsd/selective_draft/$RUN.
#SBATCH --job-name=sd-stage
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source tacc/env.sh
export RUN="${RUN:-run_$(date +%Y%m%d)}"
export OUT="${OUT:-$SCRATCH/dopsd/selective_draft/$RUN}"
echo "[slurm] job=$SLURM_JOB_ID stage=${STAGE:?set STAGE} out=$OUT config=${CONFIG:-default} max_steps=${MAX_STEPS:-default}"
bash tacc/selective_draft_smoke.sh "$STAGE"
echo "[slurm] done; sacct -j $SLURM_JOB_ID --format=JobIDRaw,Partition,Elapsed,ElapsedRaw,State,ExitCode"
