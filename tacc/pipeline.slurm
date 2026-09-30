#!/bin/bash
# One job, one queue wait: sweep -> pick best adapter -> T=0.9 traced rollouts (base + best)
# -> PI/sibling/order analysis -> figures + report. Every stage skips work that is already on disk,
# so a resubmission resumes where a killed job stopped. Submit from the repo root:
#   RUN=$SCRATCH/dopsd/runs/gsm_opsd sbatch --dependency=afterok:<train job> tacc/pipeline.slurm
# Knobs: STEPS_FILTER (sweep checkpoints), N (problems, default 300), TEMP (default 0.9),
#        DSTEPS (diffusion steps, default 128), BEST (skip the pick and use this adapter dir).
#SBATCH --job-name=dopsd-pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tacc/env.sh

RUN="${RUN:?set RUN=<training output dir>}"
N="${N:-300}"; TEMP="${TEMP:-0.9}"; DSTEPS="${DSTEPS:-128}"
ROOT="$SCRATCH/dopsd/pipeline_$(basename "$RUN")"
mkdir -p "$ROOT"
stamp() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

# 1. Sweep (greedy, paper protocol) on the same seeded subset.
stamp "sweep"
RUN="$RUN" N="$N" SUBSAMPLE="$N" OUT="$ROOT/sweep" STEPS_FILTER="${STEPS_FILTER:-448 576 1152 1344}" \
    bash tacc/eval_sweep.slurm

# 2. Best adapter by sweep accuracy (ties -> earliest step).
BEST="${BEST:-$(python - "$ROOT/sweep/sweep.csv" "$RUN" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["name"] != "base"]
best = max(rows, key=lambda r: (float(r["accuracy"]), -int(r["step"])))
print(f"{sys.argv[2]}/adapters/{best['name']}")
PY
)}"
stamp "best adapter: $BEST"; cat "$ROOT/sweep/sweep.csv"
echo "$BEST" > "$ROOT/best_adapter.txt"

# 3. Traced rollouts at TEMP for base and best (3 GPUs).
gen() {  # $1 name, $2 adapter ('' = base)
    local D="$ROOT/$1_s${DSTEPS}_t${TEMP}"
    if [[ -f "$D/.gen_done" ]]; then stamp "gen $1: done"; return; fi
    stamp "gen $1"
    local A=(); [[ -n "$2" ]] && A=(--checkpoint_path "$2")
    (cd eval && torchrun --nproc_per_node 3 --master_port 25051 eval.py --dataset gsm --subsample "$N" \
        --batch_size 8 --gen_length 256 --diffusion_steps "$DSTEPS" --block_length 32 --temperature "$TEMP" \
        --seed 42 --split test --output_dir "$D/generations" --trace_dir "$D/traces" ${A[@]+"${A[@]}"}) \
        > "$D.gen.log" 2>&1
    touch "$D/.gen_done"
}
mkdir -p "$ROOT"
gen base ""
gen best "$BEST"

# 4. Analysis, 3 shards (one per GPU); pi_sibling_analysis skips finished rollouts.
ana() {  # $1 name, $2 adapter
    local D="$ROOT/$1_s${DSTEPS}_t${TEMP}"
    stamp "analysis $1"
    local A=(); [[ -n "$2" ]] && A=(--adapter "$2")
    local pids=()
    for k in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$k python analysis/pi_sibling_analysis.py --trace_dir "$D/traces" --out_dir "$D/pisib" \
            --batch_size 16 --shard_id $k --num_shards 3 ${A[@]+"${A[@]}"} > "$D.ana$k.log" 2>&1 &
        pids+=($!)
    done
    local fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ $fail == 0 ]] || { stamp "analysis $1 FAILED (see $D.ana*.log)"; exit 1; }
    stamp "analysis $1: $(ls "$D/pisib/records" | wc -l) records"
}
ana base ""
ana best "$BEST"

# 5. Figures + report (primary run = best checkpoint; base alongside for fig4-style comparisons).
stamp "report"
FIG="$ROOT/report"
python analysis/plot_pi_sibling.py --out "$FIG" \
    --records "best=$ROOT/best_s${DSTEPS}_t${TEMP}/pisib/records" \
    --records "base=$ROOT/base_s${DSTEPS}_t${TEMP}/pisib/records" \
    --note "GSM8K, $N problems, fixed ${DSTEPS} steps, T=$TEMP; best adapter $(basename "$BEST") picked by greedy sweep; base run as second set"
python analysis/plot_pi_sibling.py --out "$FIG/base_primary" \
    --records "base=$ROOT/base_s${DSTEPS}_t${TEMP}/pisib/records" \
    --records "best=$ROOT/best_s${DSTEPS}_t${TEMP}/pisib/records" \
    --note "Same data with the BASE model as the primary run"
# Keep small, persistent copies on $WORK ($SCRATCH is purged).
mkdir -p "$DOPSD_ROOT/results/$(basename "$ROOT")"
rsync -a --include '*/' --include '*.md' --include '*.csv' --include '*.png' --include '*.txt' --exclude '*' \
    "$ROOT/" "$DOPSD_ROOT/results/$(basename "$ROOT")/"
stamp "done -> $FIG"
