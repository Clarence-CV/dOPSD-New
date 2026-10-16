#!/bin/bash
# Joint-reference OPSD. Code only: this file does not submit itself.
# First smoke: STEPS=3 N_TRAIN=8 N_EVAL=4 NPROC=1 sbatch --partition=gpu-a100-small --time=00:30:00 ...
# Before submitting, check current qlimits/SU rates and report the budget.
# Full pilot: one gpu-a100 node (3 x A100); ~5 h estimate (100 updates ~4 h + parallel MATH-500 evals ~50 min).
# Required: READOUT_SNAPSHOT=/path/to/stage1/readout/snapshot, or RESUME=/path/to/joint/checkpoint-step-N.
# Environment overrides: CONFIG, RUN, OUT, STEPS, N_TRAIN, N_EVAL, NPROC, LR, EVAL_SPLIT, SPLIT_MANIFEST.
#SBATCH --job-name=joint
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run through sbatch from the repository root}"
source tacc/env.sh
export PYTHONUNBUFFERED=1

RUN="${RUN:-joint_math}"
OUT="${OUT:-$SCRATCH/dopsd/joint/$RUN}"
CONFIG="${CONFIG:-configs/selective_draft/llada_joint_math.json}"
STEPS="${STEPS:-100}"
N_TRAIN="${N_TRAIN:-6000}"
N_EVAL="${N_EVAL:-500}"
NPROC="${NPROC:-3}"
LR="${LR:-0.0001}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
mkdir -p "$OUT"/{meta,data,train,eval}

if [[ -n "${RESUME:-}" ]]; then
    META_SNAPSHOT="$RESUME/snapshot"
    INIT=(--resume "$RESUME")
else
    : "${READOUT_SNAPSHOT:?provide the existing L28 readout snapshot}"
    META_SNAPSHOT="$READOUT_SNAPSHOT"
    INIT=(--readout-snapshot "$READOUT_SNAPSHOT")
fi
# Use exactly the model revision used to train the readout (including null if originally unpinned).
REV=$(python - "$META_SNAPSHOT/manifest.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.dumps(json.load(f)["model_revision"]))
PY
)
S=(--config "$CONFIG" --set "model.revision=$REV")
if [[ -z "${SPLIT_MANIFEST:-}" ]]; then
    python -m selective_draft.cli prepare-data "${S[@]}" --output-dir "$OUT/data"
    SPLIT_MANIFEST="$OUT/data/split_manifest.json"
fi
S+=(--set "data.split_manifest=$SPLIT_MANIFEST")
cp "$CONFIG" "$OUT/meta/config.json"
cp "$SPLIT_MANIFEST" "$OUT/meta/split_manifest.json"
cp .code_revision "$OUT/meta/git.txt" 2>/dev/null || true  # rsync copy has no .git; written by the sync step
git diff --stat >> "$OUT/meta/git.txt" 2>/dev/null || true
python -m selective_draft.cli inspect "${S[@]}" --snapshot "$META_SNAPSHOT" > "$OUT/meta/inspect.json"

torchrun --standalone --nproc_per_node="$NPROC" -m selective_draft.cli train-joint "${S[@]}" \
    --set "data.limit=$N_TRAIN" --set "training.learning_rate_guidance=$LR" \
    --max-steps "$STEPS" --save-every 25 --output-dir "$OUT/train" "${INIT[@]}"

# Identical examples, prompt, stopping/scoring, greedy sampling and threshold for all three; one GPU each, in parallel.
E=(--set "data.split=$EVAL_SPLIT" --set generation.temperature=0.0 --set generation.commit_threshold=0.9
   --set 'evaluation.commit_thresholds=[0.9]' --limit "$N_EVAL")
NG=$(nvidia-smi -L | wc -l)
ev() { local gpu=$1 name=$2; shift 2
       [[ -f "$OUT/eval/$name/summary.json" ]] && return 0
       CUDA_VISIBLE_DEVICES=$((gpu % NG)) python -m selective_draft.cli evaluate "${S[@]}" "${E[@]}" "$@" \
           --output-dir "$OUT/eval/$name" > "$OUT/eval/$name.log" 2>&1; }
ev 0 base --set guidance.mode=off &
ev 1 joint --snapshot "$OUT/train/snapshot" &
# Cheap overhead/parity control: same group readout but no injected messages.
ev 2 readout_only --snapshot "$OUT/train/snapshot" --set guidance.mode=group_readout &
FAIL=0; for j in $(jobs -p); do wait "$j" || FAIL=1; done
(( FAIL == 0 )) || { echo "an evaluation failed; see $OUT/eval/*.log"; exit 1; }

BACKUP="$WORK/inf385t/results/joint/$RUN"
mkdir -p "$BACKUP"
rsync -a --exclude references --exclude rollouts --exclude optimizer.pt "$OUT/" "$BACKUP/"
echo "Finished: $OUT"
echo "Record SU with: sacct -j $SLURM_JOB_ID --format=JobIDRaw,Partition,Elapsed,ElapsedRaw,AllocNodes,State,ExitCode"
