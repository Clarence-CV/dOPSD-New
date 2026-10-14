#!/bin/bash
# Selective Draft OPSD — MATH preliminary run, ONE job on one A100 (gpu-a100-small, 1.5 SU/node-h).
# Order = priority, so a walltime overrun still leaves the most useful outputs:
#   meta -> data -> smoke(inspect + parity) -> collect(train, analysis) -> stage 1 -> readout/teacher/consistency
#   diagnostics (training-free checks) -> stage 2 -> guided analysis states + consistency -> timed eval -> REPORT.md
# Every stage writes $OUT/.done/<stage>; a resubmission with the same RUN skips finished stages and resumes training
# from its latest checkpoint. Per-stage logs: $OUT/logs/<stage>.log; wall times: $OUT/stages.tsv.
# Critical stages (smoke, collect, both trainings) stop the job on failure; diagnostics/eval failures are recorded
# and the job continues.
#
# Submit from the repo root on a login node (logs/ must exist before sbatch):
#   mkdir -p logs && RUN=math_prelim_v1 sbatch tacc/selective_draft_math.slurm
# Budget (estimate in doc/…/results README): expected ~5.5 h; request 7:15 -> at most 10.9 SU.
# Knobs: RUN, CONFIG, N_TRAIN (80), N_ANALYSIS (64), N_EVAL (64), N_SMOKE (3), READOUT_STEPS (800),
#        GUIDE_STEPS (100), COLLECT_BS (4), EVAL_THRESHOLDS ("[0.7,0.8,0.9,0.95]"), DIAG_STATES (150).
#SBATCH --job-name=sd-math
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=07:15:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tacc/env.sh

RUN="${RUN:-math_prelim_v1}"
OUT="${OUT:-$SCRATCH/dopsd/selective_draft/$RUN}"
CONFIG="${CONFIG:-configs/selective_draft/llada_math_threshold.json}"
N_TRAIN="${N_TRAIN:-80}"; N_ANALYSIS="${N_ANALYSIS:-64}"; N_EVAL="${N_EVAL:-64}"; N_SMOKE="${N_SMOKE:-3}"
READOUT_STEPS="${READOUT_STEPS:-800}"; GUIDE_STEPS="${GUIDE_STEPS:-100}"; COLLECT_BS="${COLLECT_BS:-4}"
EVAL_THRESHOLDS="${EVAL_THRESHOLDS:-[0.7,0.8,0.9,0.95]}"; DIAG_STATES="${DIAG_STATES:-150}"
mkdir -p "$OUT"/{meta,logs,.done,states,diag,eval,smoke}
CLI="python -m selective_draft.cli"
export PYTHONUNBUFFERED=1
stamp() { echo "[sd-math $(date +%H:%M:%S)] $*"; }
[[ -f "$OUT/stages.tsv" ]] || printf "stage\tstatus\tstart\tend\twall_min\n" > "$OUT/stages.tsv"

# run_stage NAME critical|soft command...   (output tee'd to $OUT/logs/NAME.log)
run_stage() {
    local name="$1" kind="$2"; shift 2
    if [[ -f "$OUT/.done/$name" ]]; then
        stamp "$name: done earlier, skip"; printf "%s\tskip\t-\t-\t0\n" "$name" >> "$OUT/stages.tsv"; return 0
    fi
    stamp "$name: start"
    local t0=$(date +%s) s0=$(date +%F_%T) rc=0
    ( "$@" ) > >(tee -a "$OUT/logs/$name.log") 2>&1 || rc=$?
    local t1=$(date +%s)
    local wall=$(awk "BEGIN{printf \"%.1f\", ($t1-$t0)/60}")
    if [[ $rc -eq 0 ]]; then
        touch "$OUT/.done/$name"; printf "%s\tok\t%s\t%s\t%s\n" "$name" "$s0" "$(date +%F_%T)" "$wall" >> "$OUT/stages.tsv"
        stamp "$name: ok (${wall} min)"
    else
        printf "%s\tFAIL(rc=%s)\t%s\t%s\t%s\n" "$name" "$rc" "$s0" "$(date +%F_%T)" "$wall" >> "$OUT/stages.tsv"
        stamp "$name: FAILED rc=$rc (log: $OUT/logs/$name.log)"
        if [[ "$kind" == critical ]]; then python -m selective_draft.report "$OUT" > /dev/null || true; exit "$rc"; fi
    fi
}
latest_ckpt() { ls -d "$1"/checkpoint-step-* 2>/dev/null | sort -t- -k3 -n | tail -1; }

# ---------------------------------------------------------------- 0. metadata (always refreshed)
{
    echo "job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-?} node=$(hostname) start=$(date -Is)"
    echo "RUN=$RUN CONFIG=$CONFIG N_TRAIN=$N_TRAIN N_ANALYSIS=$N_ANALYSIS N_EVAL=$N_EVAL READOUT_STEPS=$READOUT_STEPS GUIDE_STEPS=$GUIDE_STEPS COLLECT_BS=$COLLECT_BS EVAL_THRESHOLDS=$EVAL_THRESHOLDS DIAG_STATES=$DIAG_STATES"
} >> "$OUT/meta/job.txt"
if [[ -f .code_revision ]]; then { echo "rsync copy of $(cat .code_revision)"; cat .code_status 2>/dev/null; } > "$OUT/meta/git.txt"
else { git rev-parse HEAD; git status --short; git diff --stat; } > "$OUT/meta/git.txt" 2>&1 || true; fi
{
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
    python -c "import sys, torch, transformers, datasets, safetensors; print('python', sys.version.split()[0], '| torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| datasets', datasets.__version__)" || true
    echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-}"
} > "$OUT/meta/env.txt" 2>&1 || true
cp "$CONFIG" "$OUT/meta/config.json"

# Pinned model revision must exist in the offline cache; otherwise use the cached main snapshot and say so.
PINNED=$(python -c "import json; print(json.load(open('$CONFIG'))['model'].get('revision') or '')")
SNAPS="$HF_HOME/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots"
if [[ -n "$PINNED" && -d "$SNAPS/$PINNED" ]]; then REV="$PINNED"
else REV=$(cat "$HF_HOME/hub/models--GSAI-ML--LLaDA-8B-Instruct/refs/main" 2>/dev/null || ls "$SNAPS" 2>/dev/null | head -1 || true); fi
[[ -n "$REV" ]] || { echo "no cached LLaDA snapshot under $SNAPS" >&2; exit 1; }
echo "pinned=$PINNED used=$REV cached=[$(ls "$SNAPS" 2>/dev/null | tr '\n' ' ')]" > "$OUT/meta/model_revision.txt"
MANIFEST="$OUT/data/split_manifest.json"
S=(--config "$CONFIG" --set "model.revision=\"$REV\"" --set "data.split_manifest=\"$MANIFEST\"")
stamp "out=$OUT rev=$REV"

# ---------------------------------------------------------------- 1. data + smoke
run_stage data critical $CLI prepare-data "${S[@]}" --output-dir "$OUT/data"
run_stage inspect critical $CLI inspect "${S[@]}"
smoke() {
    for m in off readout_only single_soft; do
        local d=$OUT/smoke/parity_$m; [[ $m == single_soft ]] && d=$OUT/smoke/parity_zero_inject
        $CLI evaluate "${S[@]}" --set data.split='"dev"' --set "guidance.mode=\"$m\"" \
            --set 'evaluation.commit_thresholds=[0.9]' --limit "$N_SMOKE" --output-dir "$d"
    done
    python - "$OUT/smoke" <<'PY'
import json, sys
o = sys.argv[1]
rows = lambda k: [json.loads(l) for l in open(f"{o}/{k}/rows_thr0.9.jsonl")]
off = rows("parity_off")
for k in ("parity_readout_only", "parity_zero_inject"):
    other = rows(k)
    bad = [a["example_id"] for a, b in zip(off, other) if a["generation"] != b["generation"] or a["row_steps"] != b["row_steps"]]
    print(k, "identical" if not bad else f"DIFFERS on {bad}")
    if bad:
        sys.exit(1)
PY
}
run_stage smoke critical smoke

# ---------------------------------------------------------------- 2. base states
# train states batched (stage-1 data only); analysis states batch 1 so counterfactual re-forwards in the
# diagnostics see exactly the recorded numerics (no padding).
run_stage collect_train critical $CLI collect "${S[@]}" --set data.split='"train"' --set guidance.mode='"off"' \
    --set "evaluation.batch_size=$COLLECT_BS" --set trace.level='"state_only"' --limit "$N_TRAIN" --output-dir "$OUT/states/train"
run_stage collect_analysis critical $CLI collect "${S[@]}" --set data.split='"analysis"' --set guidance.mode='"readout_only"' \
    --set evaluation.batch_size=1 --set trace.level='"diagnostic"' --limit "$N_ANALYSIS" --output-dir "$OUT/states/analysis_base"

# ---------------------------------------------------------------- 3. stage 1 (readout adapter)
stage1() {
    local ck; ck=$(latest_ckpt "$OUT/readout"); local R=(); [[ -n "$ck" ]] && R=(--resume "$ck")
    $CLI train-readout "${S[@]}" --set data.split='"train"' --states "$OUT/states/train" \
        --max-steps "$READOUT_STEPS" --save-every 200 --output-dir "$OUT/readout" ${R[@]+"${R[@]}"}
}
run_stage readout critical stage1

# ---------------------------------------------------------------- 4. training-free diagnostics
A=(--states "$OUT/states/analysis_base" --max-states "$DIAG_STATES")
run_stage diag_readout_native soft $CLI diagnose --kind readout "${S[@]}" "${A[@]}" --max-states 400 --output-dir "$OUT/diag/readout_native"
run_stage diag_readout_trained soft $CLI diagnose --kind readout "${S[@]}" "${A[@]}" --max-states 400 --snapshot "$OUT/readout/snapshot" --output-dir "$OUT/diag/readout_trained"
run_stage diag_teacher_L28_trained soft $CLI diagnose --kind teacher "${S[@]}" "${A[@]}" --snapshot "$OUT/readout/snapshot" --output-dir "$OUT/diag/teacher_L28_trained"
run_stage diag_teacher_L28_native soft $CLI diagnose --kind teacher "${S[@]}" "${A[@]}" --output-dir "$OUT/diag/teacher_L28_native"
run_stage diag_teacher_L24_native soft $CLI diagnose --kind teacher "${S[@]}" "${A[@]}" --set guidance.early_layer=24 --output-dir "$OUT/diag/teacher_L24_native"
run_stage diag_consistency_base soft $CLI diagnose --kind consistency "${S[@]}" "${A[@]}" --output-dir "$OUT/diag/consistency_base"
python -m selective_draft.report "$OUT" > /dev/null || true   # interim report before the long stages

# ---------------------------------------------------------------- 5. stage 2 (message injector, on-policy)
stage2() {
    local ck; ck=$(latest_ckpt "$OUT/guidance"); local R=(--readout-snapshot "$OUT/readout/snapshot")
    [[ -n "$ck" ]] && R=(--resume "$ck")
    $CLI train-guidance "${S[@]}" --set data.split='"train"' --max-steps "$GUIDE_STEPS" --save-every 25 \
        --output-dir "$OUT/guidance" "${R[@]}"
}
run_stage guidance critical stage2

# ---------------------------------------------------------------- 6. guided states on the same analysis problems
run_stage collect_analysis_guided soft $CLI collect "${S[@]}" --set data.split='"analysis"' --set guidance.mode='"single_soft"' \
    --snapshot "$OUT/guidance/snapshot" --set evaluation.batch_size=1 --set trace.level='"diagnostic"' \
    --limit "$N_ANALYSIS" --output-dir "$OUT/states/analysis_guided"
run_stage diag_consistency_guided soft $CLI diagnose --kind consistency "${S[@]}" --states "$OUT/states/analysis_guided" \
    --max-states "$DIAG_STATES" --snapshot "$OUT/guidance/snapshot" --output-dir "$OUT/diag/consistency_guided"
python -m selective_draft.report "$OUT" > /dev/null || true

# ---------------------------------------------------------------- 7. timed evaluation (dev split, batch 1)
E=(--set data.split='"dev"' --set "evaluation.commit_thresholds=$EVAL_THRESHOLDS" --limit "$N_EVAL")
run_stage eval_base soft $CLI evaluate "${S[@]}" "${E[@]}" --set guidance.mode='"off"' --output-dir "$OUT/eval/base"
run_stage eval_guided soft $CLI evaluate "${S[@]}" "${E[@]}" --set guidance.mode='"single_soft"' --snapshot "$OUT/guidance/snapshot" --output-dir "$OUT/eval/guided"
run_stage eval_readout_only soft $CLI evaluate "${S[@]}" --set data.split='"dev"' --set 'evaluation.commit_thresholds=[0.9]' --limit "$N_EVAL" \
    --set guidance.mode='"readout_only"' --snapshot "$OUT/readout/snapshot" --output-dir "$OUT/eval/readout_only"

# ---------------------------------------------------------------- 8. report + backup of the small artefacts to $WORK
python -m selective_draft.report "$OUT" > /dev/null
BK="$WORK/inf385t/results/selective_draft/$RUN"; mkdir -p "$BK"
rsync -a --exclude 'rollouts*.pt' --exclude 'rollouts' --exclude 'states' --exclude 'optimizer.pt' "$OUT/" "$BK/"
stamp "done; report $OUT/REPORT.md (backup $BK)"
echo "sacct -j ${SLURM_JOB_ID:-?} --format=JobIDRaw,Partition,Submit,Start,End,Elapsed,ElapsedRaw,AllocNodes,State,ExitCode"
