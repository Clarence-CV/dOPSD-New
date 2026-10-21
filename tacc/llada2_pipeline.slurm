#!/bin/bash
# LLaDA2.0-mini on MATH (doc/llada2-mini_engineering_plan.md). One job, one queue wait; every stage skips work
# already on disk, so resubmitting the same RUN resumes. Code only: this file does not submit itself.
# Submit from the repo root (tacc/sync_and_submit_llada2.sh does sync + qlimits + sbatch):
#   SMOKE=1 STAGES="baseline states profile readout joint jointeval" sbatch -p gpu-a100-small -t 00:45:00 tacc/llada2_pipeline.slurm
#   STAGES="baseline" sbatch tacc/llada2_pipeline.slurm
# Stages (in this order when listed):
#   baseline   base model, greedy, tau .95, block 32, cap 2048: EVAL_SPLIT[/EVAL_SUBSPLIT], sharded over the GPUs
#   states     greedy state banks: analysis:readout_val, analysis:readout_confirm, train:readout_train_initial
#   profile    all-layer logit-lens profile on readout_val -> profile/layer_decision.json (stops if no usable layer)
#   readout    train the readout at that layer, sweep checkpoints on readout_val (§7.1.2 gates + 1-SE choice),
#              open readout_confirm once for the selected checkpoint (stops if it fails the gates)
#   joint      root_success_v2 Joint training of the injector from the selected readout (JOINT_STEPS updates)
#   jointeval  dev:dev_fast (128) for base and Joint checkpoints JOINT_EVAL_STEPS (25 50 75 100), all sharded over the GPUs,
#              plus group_readout (zero-message parity/overhead) on 32; pick a checkpoint by the fixed rule (dev_fast accuracy,
#              then fewer mean steps, then earlier); run it on EVAL_SPLIT (MATH-500 test) and pair it with the baseline rows
# Knobs: RUN, CONFIG, EVAL_SPLIT (test), EVAL_SUBSPLIT (""), N_EVAL (all), READOUT_STEPS (200), JOINT_STEPS (100),
#        JOINT_EVAL_STEPS ("25 50 75 100"), SAVE_EVERY (25),
#        SMOKE=1 (tiny sizes; checks every stage end to end and prints per-example timing for the budget),
#        READOUT_SELECT=step0|<checkpoint dir> (manual readout choice when the gates fail; recorded in selected.json).
#SBATCH --job-name=llada2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=CDA26008
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run through sbatch from the repository root}"
CONDA_ENV=llada2 source tacc/env.sh
export PYTHONUNBUFFERED=1
# Job 3519705 died with a CUDA OOM at ~2.1k tokens: 31.8 GiB allocated but 6.7 GiB reserved-and-unusable
# (fragmentation from ever-growing prefix shapes). Expandable segments remove that waste.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGES="${STAGES:-baseline}"
SMOKE="${SMOKE:-0}"
RUN="${RUN:-llada2_math_v2}"
[[ "$SMOKE" == "1" ]] && RUN="${RUN}_smoke"
OUT="${OUT:-$SCRATCH/dopsd/llada2/$RUN}"
CONFIG="${CONFIG:-configs/llada2_mini_math.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_SUBSPLIT="${EVAL_SUBSPLIT:-}"
N_EVAL="${N_EVAL:-}"
READOUT_STEPS="${READOUT_STEPS:-200}"
JOINT_STEPS="${JOINT_STEPS:-100}"
JOINT_EVAL_STEPS="${JOINT_EVAL_STEPS:-25 50 75 100}"
SAVE_EVERY="${SAVE_EVERY:-25}"
DEVFAST_LIMIT=()
STATE_LIMIT=()
JOINT_EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
    N_EVAL=3; READOUT_STEPS=2; JOINT_STEPS=2; JOINT_EVAL_STEPS="1 2"; SAVE_EVERY=1; DEVFAST_LIMIT=(--limit 3)
    STATE_LIMIT=(--limit 3)
    JOINT_EXTRA=(--set training.max_rollout_attempts=2 --set data.limit=3)
fi
NG=$(nvidia-smi -L | wc -l)
# eval directory suffix: the prompt recipe (rows from different prompts never share a directory)
RECIPE=$(python -c "import json;print(json.load(open('$CONFIG'))['data']['prompt_recipe'])")
TEST_TAG="${EVAL_SPLIT}${EVAL_SUBSPLIT:+_$EVAL_SUBSPLIT}_${RECIPE}"
S=(--config "$CONFIG")
mkdir -p "$OUT"/{meta,eval,states,profile,readout,joint,logs}
stamp() { echo "[llada2 $(date +%H:%M:%S)] $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

# ---- run record (code, env, GPU) -----------------------------------------------------------------
cp "$CONFIG" "$OUT/meta/config.json"
cp .code_revision "$OUT/meta/code_revision.txt" 2>/dev/null || git rev-parse HEAD > "$OUT/meta/code_revision.txt" 2>/dev/null || true
cp .code_status "$OUT/meta/code_status.txt" 2>/dev/null || true
pip freeze > "$OUT/meta/pip_freeze.txt" 2>/dev/null || true
nvidia-smi > "$OUT/meta/nvidia_smi.txt" 2>&1 || true
echo "$SLURM_JOB_ID $(hostname) $(date -Is) STAGES=$STAGES SMOKE=$SMOKE" >> "$OUT/meta/jobs.txt"
[[ -f "$OUT/meta/inspect.json" ]] || python -m selective_draft.cli inspect "${S[@]}" > "$OUT/meta/inspect.json"

# evaluate NAME ARGS...: shard examples over all GPUs (resumable), then merge
par_eval() { local name=$1; shift
    [[ -f "$OUT/eval/$name/summary.json" ]] && { stamp "eval $name done"; return 0; }
    local pids=() i
    for ((i = 0; i < NG; i++)); do
        CUDA_VISIBLE_DEVICES=$i python -m selective_draft.cli evaluate "${S[@]}" "$@" --shard "$i/$NG" --resume \
            --output-dir "$OUT/eval/$name" > "$OUT/logs/eval_${name}_$i.log" 2>&1 & pids+=($!)
    done
    local bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done   # let surviving shards finish before failing
    (( bad == 0 )) || { echo "eval $name: a shard failed; see $OUT/logs/eval_${name}_*.log (rerun resumes)"; exit 1; }
    (( NG > 1 )) && python -m selective_draft.cli merge-eval --output-dir "$OUT/eval/$name" > "$OUT/logs/merge_$name.log"
    stamp "eval $name: $(python -c "import json;s=json.load(open('$OUT/eval/$name/summary.json'))['summaries'];print({k:{m:round(v[m],4) for m in ('n','accuracy','mean_row_steps','mean_output_tokens','truncation_rate','tpf_dinfer_mean','latency_median_s')} for k,v in s.items()})")"
}

# collect NAME SPLIT SUBSPLIT: greedy base rollouts with full state traces, sharded over the GPUs (resumable)
par_collect() { local name=$1 split=$2 sub=$3
    [[ -f "$OUT/states/$name/DONE" ]] && { stamp "states $name done"; return 0; }
    local pids=() i
    for ((i = 0; i < NG; i++)); do
        CUDA_VISIBLE_DEVICES=$i python -m selective_draft.cli collect "${S[@]}" --set "data.split=\"$split\"" \
            --set "data.subsplit=\"$sub\"" --set 'trace.level="state_only"' "${STATE_LIMIT[@]}" \
            --shard "$i/$NG" --resume --output-dir "$OUT/states/$name" > "$OUT/logs/collect_${name}_$i.log" 2>&1 & pids+=($!)
    done
    local bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done
    (( bad == 0 )) || { echo "collect $name: a shard failed; see $OUT/logs/collect_${name}_*.log (rerun resumes)"; exit 1; }
    touch "$OUT/states/$name/DONE"
    stamp "states $name: $(ls "$OUT/states/$name"/rollouts-*.pt | wc -l) rollouts"
}

latest_ckpt() { local best="" bs=-1 d s
    for d in "$1"/checkpoint-step-*; do [[ -f "$d/COMPLETE" ]] || continue; s=${d##*-}; (( s > bs )) && { bs=$s; best=$d; }; done
    echo "$best"; }
early_layer() { python -c "import json;print(json.load(open('$OUT/profile/layer_decision.json'))['early_layer'])"; }

# ---- P0 baseline ----------------------------------------------------------------------------------
if has baseline; then
    stamp "baseline on $EVAL_SPLIT${EVAL_SUBSPLIT:+:$EVAL_SUBSPLIT} (${N_EVAL:-all})"
    E=(--set "data.split=\"$EVAL_SPLIT\"" --set guidance.mode='"off"')
    [[ -n "$EVAL_SUBSPLIT" ]] && E+=(--set "data.subsplit=\"$EVAL_SUBSPLIT\"")
    [[ -n "$N_EVAL" ]] && E+=(--limit "$N_EVAL")
    par_eval "base_$TEST_TAG" "${E[@]}"
fi

# ---- stage 1: state banks, layer profile, readout -------------------------------------------------
if has states; then
    par_collect readout_val analysis readout_val
    par_collect readout_confirm analysis readout_confirm
    par_collect readout_train train readout_train_initial
fi
if has profile; then
    if [[ ! -f "$OUT/profile/layer_decision.json" ]]; then
        python -m selective_draft.cli layer-profile "${S[@]}" --states "$OUT/states/readout_val" \
            --output-dir "$OUT/profile" > "$OUT/logs/profile.log" 2>&1
    fi
    stamp "layer decision: $(cat "$OUT/profile/layer_decision.json" | tr -d '\n ' | head -c 400)"
fi
if has readout; then
    L=$(early_layer)
    [[ "$L" == "None" ]] && { echo "no usable early layer (profile/layer_decision.json); stopping"; exit 3; }
    R=(--set "guidance.early_layer=$L")
    TRAIN="$OUT/readout/train"
    CK=$(latest_ckpt "$TRAIN")
    if [[ "$CK" != *"-$READOUT_STEPS" ]]; then
        RES=(); [[ -n "$CK" ]] && RES=(--resume "$CK")
        torchrun --standalone --nproc_per_node="$NG" -m selective_draft.cli train-readout "${S[@]}" "${R[@]}" \
            --set training.readout_states_per_rollout=4 --set training.filter_correct_only=false \
            --states "$OUT/states/readout_train" --max-steps "$READOUT_STEPS" --save-every 25 \
            --output-dir "$TRAIN" "${RES[@]}" > "$OUT/logs/readout_train.log" 2>&1
    fi
    if [[ ! -f "$OUT/readout/val/readout_sweep.json" ]]; then
        python -m selective_draft.cli readout-sweep "${S[@]}" "${R[@]}" --states "$OUT/states/readout_val" \
            --checkpoints $(ls -d "$TRAIN"/checkpoint-step-*) --select --output-dir "$OUT/readout/val" \
            > "$OUT/logs/readout_val.log" 2>&1
    fi
    SEL=$(python -c "import json;d=json.load(open('$OUT/readout/val/readout_sweep.json'));s=d['selection']['selected'];print('' if s is None else (s if s=='step0' else d['checkpoints'][s]))")
    if [[ -n "${READOUT_SELECT:-}" ]]; then  # manual override (recorded): step0 or a checkpoint directory
        stamp "READOUT_SELECT=$READOUT_SELECT overrides the val rule (rule picked: ${SEL:-none})"
        SEL="$READOUT_SELECT"
    fi
    [[ -z "$SEL" ]] && { echo "no readout checkpoint passed the val gates; stopping (see readout/val)"; exit 4; }
    if [[ ! -f "$OUT/readout/confirm/readout_sweep.json" ]]; then
        CKS=(); [[ "$SEL" != "step0" ]] && CKS=("$SEL")
        python -m selective_draft.cli readout-sweep "${S[@]}" "${R[@]}" --states "$OUT/states/readout_confirm" \
            --checkpoints "${CKS[@]}" --select --output-dir "$OUT/readout/confirm" > "$OUT/logs/readout_confirm.log" 2>&1
    fi
    python - "$OUT" "$SEL" <<'PY'
import json, os, sys
out, sel = sys.argv[1], sys.argv[2]
d = json.load(open(f"{out}/readout/confirm/readout_sweep.json"))["selection"]["table"]
name = "step0" if sel == "step0" else os.path.basename(os.path.normpath(sel))
ok = d[name]["passed"]
manual = bool(os.environ.get("READOUT_SELECT"))
json.dump({"selected": sel, "confirm_passed": ok, "manual_override": manual, "confirm": d[name]},
          open(f"{out}/readout/selected.json", "w"), indent=1)
print("readout", name, "confirm passed" if ok else "NOT confirmed", "(manual override)" if manual else "")
sys.exit(0 if ok or manual else 5)
PY
fi

# ---- stage 2: Joint (root_success_v2) ---------------------------------------------------------------
if has joint; then
    L=$(early_layer)
    SEL=$(python -c "import json;print(json.load(open('$OUT/readout/selected.json'))['selected'])")
    INIT_SNAP="$OUT/readout/step0_snapshot"
    if [[ "$SEL" == "step0" ]]; then  # untrained readout = native lens; materialise it as a stage-readout snapshot
        [[ -f "$INIT_SNAP/manifest.json" ]] || python - "$CONFIG" "$L" "$INIT_SNAP" <<'PY'
import sys, torch
from selective_draft.checkpoint import save_snapshot
from selective_draft.config import Config
from selective_draft.loading import build_model_and_tokenizer
cfg = Config.load(sys.argv[1]); cfg.guidance.early_layer = int(sys.argv[2]); cfg.validate()
gm, _ = build_model_and_tokenizer(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
save_snapshot(sys.argv[3], gm, cfg, "readout", {"note": "untrained readout (step0) selected on val"})
PY
    else
        INIT_SNAP="$SEL/snapshot"
    fi
    J=(--set "guidance.early_layer=$L" --set guidance.mode='"group_soft"' --set data.split='"train"' "${JOINT_EXTRA[@]}")
    TRAIN="$OUT/joint/train"
    CK=$(latest_ckpt "$TRAIN")
    if [[ "$CK" != *"-$JOINT_STEPS" ]]; then
        INIT=(--readout-snapshot "$INIT_SNAP"); [[ -n "$CK" ]] && INIT=(--resume "$CK")
        torchrun --standalone --nproc_per_node="$NG" -m selective_draft.cli train-joint "${S[@]}" "${J[@]}" \
            --max-steps "$JOINT_STEPS" --save-every "$SAVE_EVERY" --output-dir "$TRAIN" "${INIT[@]}" > "$OUT/logs/joint_train.log" 2>&1
    fi
    stamp "joint: $(tail -1 "$TRAIN"/log_rank0.jsonl | head -c 600)"
fi
if has jointeval; then
    L=$(early_layer)
    D=(--set 'data.split="dev"' --set 'data.subsplit="dev_fast"' --set "guidance.early_layer=$L" "${DEVFAST_LIMIT[@]}")
    par_eval devfast_base "${D[@]}" --set guidance.mode='"off"'
    for st in $JOINT_EVAL_STEPS; do
        CK="$OUT/joint/train/checkpoint-step-$st"
        [[ -f "$CK/COMPLETE" ]] || { stamp "no $CK (training stopped earlier); skipped"; continue; }
        par_eval "devfast_joint_$st" "${D[@]}" --snapshot "$CK/snapshot" --set guidance.mode='"group_soft"'
    done
    LAST=$(latest_ckpt "$OUT/joint/train")
    par_eval devfast_readout_only "${D[@]}" --snapshot "$LAST/snapshot" --set guidance.mode='"group_readout"' --limit 32
    # fixed selection rule on dev_fast: accuracy, then fewer mean steps, then the earlier checkpoint
    python - "$OUT" $JOINT_EVAL_STEPS <<'PY'
import json, os, sys
out, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
rows = []
for st in steps:
    p = f"{out}/eval/devfast_joint_{st}/summary.json"
    if os.path.exists(p):
        s = json.load(open(p))["summaries"]
        s = next(iter(s.values()))
        rows.append({"step": st, "accuracy": s["accuracy"], "mean_row_steps": s["mean_row_steps"],
                     "latency_median_s": s["latency_median_s"]})
base = next(iter(json.load(open(f"{out}/eval/devfast_base/summary.json"))["summaries"].values()))
if rows:
    best = min(rows, key=lambda r: (-r["accuracy"], r["mean_row_steps"], r["step"]))
    ck = f"{out}/joint/train/checkpoint-step-{best['step']}"
else:  # training ended before the first evaluated checkpoint: fall back to the last complete one (recorded)
    cks = [d for d in os.listdir(f"{out}/joint/train") if d.startswith("checkpoint-step-")
           and os.path.exists(f"{out}/joint/train/{d}/COMPLETE")]
    last = max(cks, key=lambda d: int(d.rsplit("-", 1)[1]))
    best, ck = {"step": int(last.rsplit("-", 1)[1]), "fallback": "no evaluated candidate"}, f"{out}/joint/train/{last}"
sel = {"rule": "max dev_fast accuracy, then min mean_row_steps, then earliest step", "selected_step": best["step"],
       "checkpoint": ck, "candidates": rows, "fallback": best.get("fallback"),
       "base": {k: base[k] for k in ("accuracy", "mean_row_steps", "latency_median_s")}}
json.dump(sel, open(f"{out}/eval/joint_selection.json", "w"), indent=1)
print("selected joint step", best["step"], rows, "base", sel["base"])
PY
    SELCK=$(python -c "import json;print(json.load(open('$OUT/eval/joint_selection.json'))['checkpoint'])")
    T=(--set "data.split=\"$EVAL_SPLIT\"" --set "guidance.early_layer=$L")
    [[ -n "$EVAL_SUBSPLIT" ]] && T+=(--set "data.subsplit=\"$EVAL_SUBSPLIT\"")
    [[ -n "$N_EVAL" ]] && T+=(--limit "$N_EVAL")
    BASE_NAME="base_$TEST_TAG"
    JOINT_NAME="joint_selected_$TEST_TAG"
    par_eval "$BASE_NAME" "${T[@]}" --set guidance.mode='"off"'   # no-op when the baseline stage already ran
    par_eval "$JOINT_NAME" "${T[@]}" --snapshot "$SELCK/snapshot" --set guidance.mode='"group_soft"'
    python - "$OUT/eval/$BASE_NAME" "$OUT/eval/$JOINT_NAME" "$OUT/eval/paired_$TEST_TAG.json" <<'PY'
import glob, json, sys
def rows(d):
    p = sorted(glob.glob(f"{d}/rows_*.jsonl"))
    p = [x for x in p if ".shard" not in x][0]
    return {r["example_id"]: r for r in map(json.loads, open(p))}
b, j = rows(sys.argv[1]), rows(sys.argv[2])
ids = sorted(set(b) & set(j))
n = len(ids)
fix = sum(j[i]["is_correct"] and not b[i]["is_correct"] for i in ids)
brk = sum(b[i]["is_correct"] and not j[i]["is_correct"] for i in ids)
m = lambda d, k: sum(d[i][k] for i in ids) / n
out = {"n": n, "base_acc": m(b, "is_correct"), "joint_acc": m(j, "is_correct"), "fixed": fix, "broken": brk,
       "base_mean_steps": m(b, "row_steps"), "joint_mean_steps": m(j, "row_steps"),
       "base_mean_latency_s": m(b, "latency_s"), "joint_mean_latency_s": m(j, "latency_s"),
       "base_tpf_dinfer_mean": m(b, "tpf_dinfer"), "joint_tpf_dinfer_mean": m(j, "tpf_dinfer"),
       "base_mean_nfe_dinfer": m(b, "nfe_dinfer"), "joint_mean_nfe_dinfer": m(j, "nfe_dinfer"),
       "base_truncation": sum(b[i]["stop_reason"] == "length" for i in ids) / n,
       "joint_truncation": sum(j[i]["stop_reason"] == "length" for i in ids) / n}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("paired", json.dumps(out))
PY
fi

BACKUP="$WORK/inf385t/results/llada2/$RUN"
mkdir -p "$BACKUP"
rsync -a --exclude 'states/' --exclude 'references/' --exclude 'rollouts/' --exclude 'optimizer.pt' "$OUT/" "$BACKUP/"
stamp "finished: $OUT (backup $BACKUP)"
echo "Record SU with: sacct -j $SLURM_JOB_ID --format=JobIDRaw,Partition,Elapsed,ElapsedRaw,AllocNodes,State,ExitCode"
