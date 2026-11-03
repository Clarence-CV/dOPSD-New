#!/bin/bash
# Sibling OPSD on LLaDA2.0-mini (doc/plan_a_loo_defer.md). One job, one queue wait; every stage skips work already on
# disk, so resubmitting the same RUN resumes. Written for Vista (1 GH200 96 GB per node); also runs on Lonestar6.
# Submit from the repo root (tacc/sync_and_submit_llada2.sh with SCRIPT=tacc/sibling_pipeline.slurm does sync + sbatch):
#   dev check:  TACC_HOST=vista SCRIPT=tacc/sibling_pipeline.slurm DEV=1 TESTS=1 RUN=sib_dev \
#               SBATCH_ARGS="-p gh-dev -t 02:00:00" bash tacc/sync_and_submit_llada2.sh
#   main run:   TACC_HOST=vista SCRIPT=tacc/sibling_pipeline.slurm RUN=sib_v1 SBATCH_ARGS="-p gh -t 40:00:00" ...
# Stages (STAGES, in this order):
#   memprobe  peak memory / time of one training state at N = 1024/2048/2336 (synthetic), Teacher batch 4 and 8
#   diag   Phase 0 on dev:dev_fast (first DIAG_N): base greedy accuracy at tau 0.95 and 0.7; sibling-diag rollouts
#          (T=0.9, up to 8 attempts) at tau 0.7 and 0.95 with frozen-base leave-one-out statistics
#   train  train-sibling: TRAIN_STEPS updates, QPU questions per GPU per update, checkpoint every SAVE_EVERY
#   eval   dev_fast (first DEV_N) at tau 0.7: base and checkpoints EVAL_STEPS; pick by accuracy, then fewer steps,
#          then earlier; FINAL_TEST=1: MATH-500 test, base vs selected, at tau 0.7 and 0.95, paired summary
# Knobs: RUN, CONFIG, DIAG_N (64), DEV_N (128), TRAIN_STEPS (400), TPG (training processes per GPU, 2; >1 uses gloo),
#        QPU (questions per process per update; 2 when NG*TPG < 3, else 1), SAVE_EVERY (50),
#        EVAL_STEPS ("100 200 300 400"), FINAL_TEST (1), N_TEST (all 500), TESTS=1 (CPU tests first), DEV=1 (small
#        sizes for the 2-hour dev queue), MOE_IMPL (override model.moe_impl), PPG (evaluation/diag processes per GPU, 1),
#        LOO_BATCH (Teacher rows per forward, 4), DIAG_TAUS ("0.7 0.95"), DIAG_EVAL_TAUS ("0.95 0.7"), DIAG_K (8).
# Monitoring: nvidia-smi samples (utilisation, memory, power) every 30 s -> $OUT/logs/gpu_monitor_<job>.csv;
#        train/log_rank0.jsonl has per-update loss terms, LOO statistics, on-policy accuracy/TPF, timing and peak memory.
#SBATCH --job-name=sibling
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run through sbatch from the repository root}"
CONDA_ENV=llada2 source tacc/env.sh
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Stage the weights on node-local disk once per job (see tacc/llada2_pipeline.slurm for the history).
MODEL_CACHE="models--inclusionAI--LLaDA2.0-mini"
MODEL_HUB_SRC="${MODEL_HUB_SRC:-$SCRATCH/hf_hub}"
[[ -d "$MODEL_HUB_SRC/$MODEL_CACHE/snapshots" ]] || MODEL_HUB_SRC="$HF_HOME/hub"
if [[ -d "$MODEL_HUB_SRC/$MODEL_CACHE" ]]; then
    export HF_HUB_CACHE="/tmp/$USER-hf-hub-${SLURM_JOB_ID:-local}"
    mkdir -p "$HF_HUB_CACHE"
    trap 'rm -rf "$HF_HUB_CACHE"' EXIT
    t0=$(date +%s)
    timeout 1800 cp -a "$MODEL_HUB_SRC/$MODEL_CACHE" "$HF_HUB_CACHE/" \
        || { echo "copying the model to node-local /tmp failed or took over 30 min"; exit 8; }
    echo "[sibling] model staged from $MODEL_HUB_SRC in $(( $(date +%s) - t0 )) s"
fi

STAGES="${STAGES:-memprobe diag train eval}"
RUN="${RUN:-sib_v1}"
OUT="${OUT:-$SCRATCH/dopsd/sibling/$RUN}"
CONFIG="${CONFIG:-configs/llada2_mini_sibling.json}"
NG=$(nvidia-smi -L | wc -l)
DIAG_N="${DIAG_N:-64}"
DEV_N="${DEV_N:-128}"
TRAIN_STEPS="${TRAIN_STEPS:-400}"
TPG="${TPG:-2}"                                                     # training processes per GPU (GH200 96 GB: 2 x ~42 GB)
NTR=$((NG * TPG))
QPU="${QPU:-$(( NTR >= 3 ? 1 : 2 ))}"                               # questions per process per update
SAVE_EVERY="${SAVE_EVERY:-50}"
EVAL_STEPS="${EVAL_STEPS:-100 200 300 400}"
FINAL_TEST="${FINAL_TEST:-1}"
N_TEST="${N_TEST:-}"
TESTS="${TESTS:-0}"
PPG="${PPG:-1}"
DIAG_TAUS="${DIAG_TAUS:-0.7 0.95}"             # sibling-diag thresholds
DIAG_EVAL_TAUS="${DIAG_EVAL_TAUS:-0.95 0.7}"   # greedy base evaluations in the diag stage
DIAG_K="${DIAG_K:-8}"                          # rollout attempts per question in sibling-diag
LOO_BATCH="${LOO_BATCH:-4}"
if [[ "${DEV:-0}" == "1" ]]; then
    DIAG_N="${DIAG_N_DEV:-12}"; DEV_N=8; TRAIN_STEPS=4; QPU=1; SAVE_EVERY=2; EVAL_STEPS="2 4"; FINAL_TEST=0
    PPG="${PPG_DEV:-2}"
fi
S=(--config "$CONFIG" --set "training.loo_batch_size=$LOO_BATCH")
[[ -n "${MOE_IMPL:-}" ]] && S+=(--set "model.moe_impl=\"$MOE_IMPL\"")
BASE=(--set sibling.lora_rank=0)                                    # frozen base, no adapters
T95=(--set generation.commit_threshold=0.95)
mkdir -p "$OUT"/{meta,eval,diag,train,logs,memprobe}
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "$OUT/logs/gpu_monitor_${SLURM_JOB_ID:-local}.csv" 2>/dev/null &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null; [[ -n "${HF_HUB_CACHE:-}" && "$HF_HUB_CACHE" == /tmp/* ]] && rm -rf "$HF_HUB_CACHE"' EXIT
stamp() { echo "[sibling $(date +%H:%M:%S)] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if [[ "$TESTS" == "1" ]]; then
    for t in sibling decoder guidance training data; do python tests/test_$t.py > "$OUT/logs/test_$t.log" 2>&1 || { echo "CPU test $t FAILED"; tail -20 "$OUT/logs/test_$t.log"; exit 6; }; done
    stamp "CPU tests passed"
fi
cp "$CONFIG" "$OUT/meta/config.json"
cp .code_revision "$OUT/meta/code_revision.txt" 2>/dev/null || true
cp .code_status "$OUT/meta/code_status.txt" 2>/dev/null || true
nvidia-smi > "$OUT/meta/nvidia_smi.txt" 2>&1 || true
echo "$SLURM_JOB_ID $(hostname) $(date -Is) STAGES=$STAGES DEV=${DEV:-0} NG=$NG TPG=$TPG QPU=$QPU PPG=$PPG" >> "$OUT/meta/jobs.txt"
[[ -s "$OUT/meta/inspect.json" ]] || timeout 900 python -m selective_draft.cli inspect "${S[@]}" > "$OUT/meta/inspect.json" \
    || { rm -f "$OUT/meta/inspect.json"; echo "model load/inspect failed or took over 15 min"; exit 7; }

summ() { python -c "import json;s=json.load(open('$1/summary.json'))['summaries'];print({k:{m:round(v[m],4) for m in ('n','accuracy','mean_row_steps','mean_output_tokens','truncation_rate','tpf_dinfer_mean','latency_median_s')} for k,v in s.items()})"; }
par_eval() { local name=$1; shift
    [[ -f "$OUT/eval/$name/summary.json" ]] && { stamp "eval $name done: $(summ "$OUT/eval/$name")"; return 0; }
    local pids=() i NS=$((NG * PPG))
    for ((i = 0; i < NS; i++)); do
        CUDA_VISIBLE_DEVICES=$((i % NG)) python -m selective_draft.cli evaluate "${S[@]}" "$@" --shard "$i/$NS" --resume \
            --output-dir "$OUT/eval/$name" > "$OUT/logs/eval_${name}_$i.log" 2>&1 & pids+=($!)
    done
    local bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done
    (( bad == 0 )) || { echo "eval $name: a shard failed; see $OUT/logs/eval_${name}_*.log (rerun resumes)"; exit 1; }
    (( NS > 1 )) && python -m selective_draft.cli merge-eval --output-dir "$OUT/eval/$name" > "$OUT/logs/merge_$name.log"
    stamp "eval $name: $(summ "$OUT/eval/$name")"
}
par_diag() { local name=$1; shift
    [[ -f "$OUT/diag/$name/summary.json" ]] && { stamp "diag $name done"; return 0; }
    local pids=() i NS=$((NG * PPG))
    for ((i = 0; i < NS; i++)); do
        CUDA_VISIBLE_DEVICES=$((i % NG)) python -m selective_draft.cli sibling-diag "${S[@]}" "${BASE[@]}" "$@" --shard "$i/$NS" \
            --output-dir "$OUT/diag/$name" > "$OUT/logs/diag_${name}_$i.log" 2>&1 & pids+=($!)
    done
    local bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done
    (( bad == 0 )) || { echo "diag $name failed; see $OUT/logs/diag_${name}_*.log"; exit 1; }
    python -m selective_draft.cli sibling-diag "${S[@]}" "${BASE[@]}" "$@" --summarize-only --output-dir "$OUT/diag/$name" \
        > "$OUT/logs/diag_${name}_summary.log"
    stamp "diag $name: $(python -c "
import json;s=json.load(open('$OUT/diag/$name/summary.json'))
f=lambda d:{k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items()}
print(f({k:s[k] for k in ('n_questions','tau','first_attempt_accuracy','solved_within_budget','mean_attempts_to_success','mean_rollout_seconds')}))
for g in ('correct','wrong'): print(g, f(s[g]))")"
}
latest_ckpt() { local best="" bs=-1 d s
    for d in "$1"/checkpoint-step-*; do [[ -f "$d/COMPLETE" ]] || continue; s=${d##*-}; (( s > bs )) && { bs=$s; best=$d; }; done
    echo "$best"; }
DEVSET=(--set 'data.split="dev"' --set 'data.subsplit="dev_fast"')

# ---- memory / speed probe of one training state ---------------------------------------------------------------------
if has memprobe; then
    [[ -f "$OUT/memprobe/sibling_memprobe.json" ]] || CUDA_VISIBLE_DEVICES=0 python -m selective_draft.cli sibling-memprobe \
        "${S[@]}" --output-dir "$OUT/memprobe" > "$OUT/logs/memprobe.log" 2>&1 \
        || { echo "memprobe failed (continuing); see $OUT/logs/memprobe.log"; tail -30 "$OUT/logs/memprobe.log"; }
    stamp "memprobe: $(grep '^{' "$OUT/logs/memprobe.log" | tr '\n' ' ' | head -c 1500)"
fi

# ---- Phase 0: diagnostics on the frozen base --------------------------------------------------------------------
if has diag; then
    for tau in $DIAG_EVAL_TAUS; do   # greedy base accuracy / TPF reference
        par_eval "diag_base_t${tau#0.}" "${DEVSET[@]}" "${BASE[@]}" --set "generation.commit_threshold=$tau" --limit "$DIAG_N"
    done
    for tau in $DIAG_TAUS; do        # T_train rollouts + frozen-base LOO statistics
        par_diag "tau${tau#0.}" "${DEVSET[@]}" --set "generation.commit_threshold=$tau" \
            --set "training.max_rollout_attempts=$DIAG_K" --limit "$DIAG_N"
    done
fi

# ---- Phase 1: training -------------------------------------------------------------------------------------------
if has train; then
    TRAIN="$OUT/train"
    CK=$(latest_ckpt "$TRAIN")
    if [[ "$CK" != *"-$TRAIN_STEPS" ]]; then
        RES=(); [[ -n "$CK" ]] && RES=(--resume "$CK")
        BK=(); (( TPG > 1 )) && BK=(env DOPSD_DIST_BACKEND=gloo)
        "${BK[@]}" torchrun --standalone --nproc_per_node="$NTR" -m selective_draft.cli train-sibling "${S[@]}" \
            --set 'data.split="train"' --set "sibling.questions_per_update=$QPU" \
            --max-steps "$TRAIN_STEPS" --save-every "$SAVE_EVERY" --output-dir "$TRAIN" "${RES[@]}" \
            > "$OUT/logs/train.log" 2>&1
    fi
    stamp "train: $(tail -1 "$TRAIN"/log_rank0.jsonl | head -c 900)"
fi

# ---- evaluation -----------------------------------------------------------------------------------------------------
if has eval; then
    D=("${DEVSET[@]}" --limit "$DEV_N")
    par_eval devfast_base_t70 "${D[@]}" "${BASE[@]}"
    for st in $EVAL_STEPS; do
        CK="$OUT/train/checkpoint-step-$st"
        [[ -f "$CK/COMPLETE" ]] || { stamp "no $CK; skipped"; continue; }
        par_eval "devfast_step${st}_t70" "${D[@]}" --snapshot "$CK/snapshot"
    done
    python - "$OUT" $EVAL_STEPS <<'PY'
import json, os, sys
out, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
first = lambda p: next(iter(json.load(open(p))["summaries"].values()))
rows = []
for st in steps:
    p = f"{out}/eval/devfast_step{st}_t70/summary.json"
    if os.path.exists(p):
        s = first(p)
        rows.append({"step": st, "accuracy": s["accuracy"], "mean_row_steps": s["mean_row_steps"],
                     "tpf_dinfer_mean": s["tpf_dinfer_mean"]})
base = first(f"{out}/eval/devfast_base_t70/summary.json")
best = min(rows, key=lambda r: (-r["accuracy"], r["mean_row_steps"], r["step"])) if rows else None
sel = {"rule": "max dev_fast accuracy at tau 0.7, then min mean_row_steps, then earliest step",
       "selected_step": best and best["step"], "checkpoint": best and f"{out}/train/checkpoint-step-{best['step']}",
       "candidates": rows, "base": {k: base[k] for k in ("accuracy", "mean_row_steps", "tpf_dinfer_mean")}}
json.dump(sel, open(f"{out}/eval/selection.json", "w"), indent=1)
print("selected", sel["selected_step"], rows, "base", sel["base"])
PY
    SELCK=$(python -c "import json;print(json.load(open('$OUT/eval/selection.json'))['checkpoint'] or '')")
    if [[ "$FINAL_TEST" == "1" && -n "$SELCK" ]]; then
        T=(--set 'data.split="test"'); [[ -n "$N_TEST" ]] && T+=(--limit "$N_TEST")
        par_eval test_base_t70 "${T[@]}" "${BASE[@]}"
        par_eval test_sel_t70 "${T[@]}" --snapshot "$SELCK/snapshot"
        par_eval test_base_t95 "${T[@]}" "${BASE[@]}" "${T95[@]}"
        par_eval test_sel_t95 "${T[@]}" "${T95[@]}" --snapshot "$SELCK/snapshot"
        python - "$OUT/eval" <<'PY'
import glob, json, sys
ev = sys.argv[1]
def rows(d):
    p = [x for x in sorted(glob.glob(f"{ev}/{d}/rows_*.jsonl")) if ".shard" not in x][0]
    return {r["example_id"]: r for r in map(json.loads, open(p))}
res = {}
for t in ("t70", "t95"):
    b, s = rows(f"test_base_{t}"), rows(f"test_sel_{t}")
    ids = sorted(set(b) & set(s)); n = len(ids)
    m = lambda d, k: sum(d[i][k] for i in ids) / n
    res[t] = {"n": n, "base_acc": m(b, "is_correct"), "sel_acc": m(s, "is_correct"),
              "fixed": sum(s[i]["is_correct"] and not b[i]["is_correct"] for i in ids),
              "broken": sum(b[i]["is_correct"] and not s[i]["is_correct"] for i in ids),
              "base_tpf": m(b, "tpf_dinfer"), "sel_tpf": m(s, "tpf_dinfer"),
              "base_steps": m(b, "row_steps"), "sel_steps": m(s, "row_steps"),
              "base_latency_s": m(b, "latency_s"), "sel_latency_s": m(s, "latency_s")}
json.dump(res, open(f"{ev}/paired_test.json", "w"), indent=1)
print("paired", json.dumps(res))
PY
    fi
fi

BACKUP="$WORK/inf385t/results/sibling/$RUN"
mkdir -p "$BACKUP"
rsync -a --exclude 'optimizer.pt' "$OUT/" "$BACKUP/"
stamp "finished: $OUT (backup $BACKUP)"
echo "Record SU with: sacct -j $SLURM_JOB_ID --format=JobIDRaw,Partition,Elapsed,ElapsedRaw,AllocNodes,State,ExitCode"
