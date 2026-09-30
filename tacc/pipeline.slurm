#!/bin/bash
# One job, one queue wait: greedy sweep -> pick best adapter -> traced rollouts at TEMP for
# {base, best} x {fixed budget, confidence threshold} -> PI/sibling/order analysis records.
# Figures/reports are made offline from the records (CPU only; no GPU-node time).
# Every stage skips work already on disk, so a resubmission resumes. Submit from the repo root:
#   RUN=$SCRATCH/dopsd/runs/gsm_opsd sbatch [--dependency=afterok:<train job>] tacc/pipeline.slurm
# Knobs: STEPS_FILTER (sweep checkpoints), N (problems, default 300), TEMP (default 0.9),
#        DECODES (default "s128 thr0.9": sN = fixed budget N steps, thrX = threshold X),
#        BEST (skip the pick and use this adapter dir).
#SBATCH --job-name=dopsd-pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tacc/env.sh

RUN="${RUN:?set RUN=<training output dir>}"
N="${N:-300}"; TEMP="${TEMP:-0.9}"; DECODES="${DECODES:-s128 thr0.9}"
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

# 3. Traced rollouts at TEMP for base and best under each decoding (3 GPUs).
dec_args() {  # $1 = sN | thrX
    case "$1" in
        s*)   echo "--diffusion_steps ${1#s}" ;;
        thr*) echo "--diffusion_steps 128 --threshold ${1#thr}" ;;
        *)    echo "unknown decode $1" >&2; exit 1 ;;
    esac
}
gen() {  # $1 model name, $2 adapter ('' = base), $3 decode
    local D="$ROOT/$1_$3_t${TEMP}"
    if [[ -f "$D/.gen_done" ]]; then stamp "gen $1 $3: done"; return; fi
    stamp "gen $1 $3"
    local A=(); [[ -n "$2" ]] && A=(--checkpoint_path "$2")
    (cd eval && torchrun --nproc_per_node 3 --master_port 25051 eval.py --dataset gsm --subsample "$N" \
        --batch_size 8 --gen_length 256 --block_length 32 $(dec_args "$3") --temperature "$TEMP" \
        --seed 42 --split test --output_dir "$D/generations" --trace_dir "$D/traces" ${A[@]+"${A[@]}"}) \
        > "$D.gen.log" 2>&1
    touch "$D/.gen_done"
}
for dec in $DECODES; do
    gen base "" "$dec"
    gen best "$BEST" "$dec"
done

# 4. Analysis, 3 shards (one per GPU); pi_sibling_analysis skips finished rollouts.
ana() {  # $1 model name, $2 adapter, $3 decode
    local D="$ROOT/$1_$3_t${TEMP}"
    stamp "analysis $1 $3"
    local A=(); [[ -n "$2" ]] && A=(--adapter "$2")
    local pids=()
    for k in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$k python analysis/pi_sibling_analysis.py --trace_dir "$D/traces" --out_dir "$D/pisib" \
            --batch_size 16 --shard_id $k --num_shards 3 ${A[@]+"${A[@]}"} > "$D.ana$k.log" 2>&1 &
        pids+=($!)
    done
    local fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ $fail == 0 ]] || { stamp "analysis $1 $3 FAILED (see $D.ana*.log)"; exit 1; }
    stamp "analysis $1 $3: $(ls "$D/pisib/records" | wc -l) records"
}
for dec in $DECODES; do
    ana base "" "$dec"
    ana best "$BEST" "$dec"
done

# 5. Small persistent copies on $WORK ($SCRATCH is purged); records are pulled for offline plotting.
mkdir -p "$DOPSD_ROOT/results/$(basename "$ROOT")"
rsync -a --include '*/' --include '*.csv' --include '*.txt' --include '*.log' --exclude '*' \
    "$ROOT/" "$DOPSD_ROOT/results/$(basename "$ROOT")/"
stamp "done -> $ROOT"
