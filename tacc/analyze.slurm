#!/bin/bash
# PI-gain / sibling-dependency analysis over a trace dir (analysis/pi_sibling_analysis.py).
#   TRACES=$SCRATCH/dopsd/eval/gsm_base_s128_t0.0/traces sbatch tacc/analyze.slurm
#   TRACES=... ADAPTER=$SCRATCH/dopsd/runs/gsm_opsd/adapters/step-1344 sbatch tacc/analyze.slurm
# Optional: NGPU=1 (with -p gpu-a100-small), PI_SAMPLES=K, TEACHER=self, MAX_ROLLOUTS=N, OUT=dir.
#SBATCH --job-name=dopsd-analyze
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

TRACES="${TRACES:?set TRACES=<trace dir>}"
OUT="${OUT:-$(dirname "$TRACES")/pisib${ADAPTER:+_$(basename "$ADAPTER")}}"
NGPU="${NGPU:-3}"
ARGS=(--trace_dir "$TRACES" --out_dir "$OUT" --pi_samples "${PI_SAMPLES:-1}" --teacher "${TEACHER:-fixed}"
      --max_rollouts "${MAX_ROLLOUTS:-0}" --batch_size "${BATCH:-8}")
[[ -n "${ADAPTER:-}" ]] && ARGS+=(--adapter "$ADAPTER")
echo "[slurm] job=$SLURM_JOB_ID traces=$TRACES adapter=${ADAPTER:-none} ngpu=$NGPU -> $OUT"

pids=()
for ((k = 0; k < NGPU; k++)); do
    CUDA_VISIBLE_DEVICES=$k python analysis/pi_sibling_analysis.py "${ARGS[@]}" --shard_id $k --num_shards "$NGPU" \
        > "logs/analyze-$SLURM_JOB_ID-shard$k.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "[slurm] records: $(ls "$OUT/records" | wc -l)  ($(du -sh "$OUT" | cut -f1))"
exit $fail
