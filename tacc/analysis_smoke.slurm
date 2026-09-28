#!/bin/bash
# End-to-end check of the analysis pipeline on ONE A100 (gpu-a100-small, 1.5 SU/h):
# base LLaDA, N GSM8K problems (N=8 by default: one full eval batch, so left padding and
# per-row grading are exercised, and both correct and wrong rollouts almost surely appear),
# traces at 128 steps (|C_t|=2) and 64 steps (|C_t|=4),
# then PI/sibling analysis on both. Submit from the repo root: sbatch tacc/analysis_smoke.slurm
# MODES picks decoding settings: sN = fixed budget with N steps, thrX = confidence threshold X
#   (e.g. MODES="thr0.9" sbatch tacc/analysis_smoke.slurm). Default: "s128 s64".
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

for MODE in ${MODES:-s128 s64}; do
    D="$OUT/$MODE"
    rm -rf "$D"
    case "$MODE" in
        s*)   DEC=(--diffusion_steps "${MODE#s}") ;;
        thr*) DEC=(--diffusion_steps 128 --threshold "${MODE#thr}") ;;
        *)    echo "unknown mode $MODE"; exit 1 ;;
    esac
    t0=$(date +%s)
    (cd eval && torchrun --nproc_per_node 1 --master_port 25031 eval.py --dataset gsm --subsample "${N:-8}" \
        --batch_size 8 --gen_length 256 --block_length 32 --temperature 0.0 "${DEC[@]}" \
        --output_dir "$D/generations" --trace_dir "$D/traces")
    t1=$(date +%s)
    python analysis/pi_sibling_analysis.py --trace_dir "$D/traces" --out_dir "$D/pisib" --batch_size "${BATCH:-8}"
    t2=$(date +%s)
    echo "[anasmoke] mode=$MODE  generate+trace $((t1 - t0))s   analysis $((t2 - t1))s"
    python analysis/summarize_traces.py "$D/traces"
done
du -sh "$OUT"/*/pisib 2>/dev/null
