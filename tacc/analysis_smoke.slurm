#!/bin/bash
# End-to-end check of the analysis pipeline on ONE A100 (gpu-a100-small, 1.5 SU/h):
# base LLaDA, 20 GSM8K problems, traces at 128 steps (|C_t|=2) and 64 steps (|C_t|=4),
# then PI/sibling analysis on both. Submit from the repo root: sbatch tacc/analysis_smoke.slurm
#SBATCH --job-name=dopsd-anasmoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tacc/env.sh
OUT="$SCRATCH/dopsd/anasmoke"
rm -rf "$OUT"

for STEPS in 128 64; do
    D="$OUT/s$STEPS"
    t0=$(date +%s)
    (cd eval && torchrun --nproc_per_node 1 --master_port 25031 eval.py --dataset gsm --subsample 20 \
        --batch_size 8 --gen_length 256 --diffusion_steps $STEPS --block_length 32 --temperature 0.0 \
        --output_dir "$D/generations" --trace_dir "$D/traces")
    t1=$(date +%s)
    python analysis/pi_sibling_analysis.py --trace_dir "$D/traces" --out_dir "$D/pisib" --batch_size "${BATCH:-8}"
    t2=$(date +%s)
    echo "[anasmoke] steps=$STEPS  generate+trace $((t1 - t0))s   analysis $((t2 - t1))s"
    python analysis/summarize_traces.py "$D/traces"
done
du -sh "$OUT"/*/pisib
