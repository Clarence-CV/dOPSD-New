#!/bin/bash
# Screen many LoRA snapshots in ONE job (one queue wait): paper protocol (greedy, gen 256, 128 steps,
# block 32) on a seeded SUBSAMPLE of GSM8K test, 3 GPUs via torchrun, no traces.
#   RUN=$SCRATCH/dopsd/runs/gsm_opsd sbatch tacc/eval_sweep.slurm
# Optional: SUBSAMPLE (default 300), STEPS_FILTER="64 128 ..." (default: every adapters/step-*), BASE=1
# (also evaluate the base model on the same subset). Writes $OUT/sweep.csv: name,step,n,correct,accuracy.
#SBATCH --job-name=dopsd-sweep
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

RUN="${RUN:?set RUN=<training output dir>}"
N="${SUBSAMPLE:-300}"
OUT="${OUT:-$SCRATCH/dopsd/eval/sweep_$(basename "$RUN")_n$N}"
mkdir -p "$OUT"
CKPTS=()
[[ "${BASE:-1}" == "1" ]] && CKPTS+=(base)
if [[ -n "${STEPS_FILTER:-}" ]]; then
    for s in $STEPS_FILTER; do CKPTS+=("$RUN/adapters/step-$s"); done
else
    while read -r d; do CKPTS+=("$d"); done < <(ls -d "$RUN"/adapters/step-* | sort -t- -k2 -n)
fi
echo "[sweep] job=$SLURM_JOB_ID n=$N ckpts=${#CKPTS[@]} -> $OUT"
[[ -f "$OUT/sweep.csv" ]] || echo "name,step,n,correct,accuracy,seconds" > "$OUT/sweep.csv"

for C in "${CKPTS[@]}"; do
    name=$([[ "$C" == base ]] && echo base || basename "$C")
    step=$([[ "$C" == base ]] && echo 0 || echo "${name#step-}")
    if grep -q "^$name," "$OUT/sweep.csv"; then echo "[sweep] skip $name (done)"; continue; fi
    GEN="$OUT/gen_$name"
    ARGS=(--dataset gsm --subsample "$N" --batch_size 8 --gen_length 256 --diffusion_steps 128 --block_length 32
          --temperature 0.0 --remasking low_confidence --seed 42 --split test --output_dir "$GEN")
    [[ "$C" != base ]] && ARGS+=(--checkpoint_path "$C")
    t0=$(date +%s)
    (cd eval && torchrun --nproc_per_node 3 --master_port 25041 eval.py "${ARGS[@]}") > "$OUT/log_$name.txt" 2>&1
    # accuracy from the per-row grades eval.py writes (same parser as parse_and_get_acc for gsm)
    python - "$GEN" "$name" "$step" "$(( $(date +%s) - t0 ))" >> "$OUT/sweep.csv" <<'PY'
import glob, json, sys
gen, name, step, secs = sys.argv[1:]
rows = [g for f in glob.glob(f"{gen}/**/*_generations.json", recursive=True) for g in json.load(open(f))["generations"]]
ok = sum(bool(r["is_correct"]) for r in rows)
print(f"{name},{step},{len(rows)},{ok},{ok / max(len(rows), 1):.4f},{secs}")
PY
    # official parser too (its gsm parser is separate code; keep both for cross-checking)
    (cd eval && python parse_and_get_acc.py "$(dirname "$(find "$GEN" -name '*_generations.json' | head -1)")" gsm) \
        >> "$OUT/log_$name.txt" 2>&1 || true
    tail -1 "$OUT/sweep.csv"
done
echo "[sweep] done"; cat "$OUT/sweep.csv"
