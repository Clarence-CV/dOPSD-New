#!/bin/bash
# Staged real-model smoke for the Selective Draft pipeline. Run INSIDE an idev session (gpu-a100-dev or
# gpu-a100-small), from the repo root. Each stage is small; nothing here starts a long training run.
#   bash tacc/selective_draft_smoke.sh env       # python/torch/CUDA, offline caches, import check
#   bash tacc/selective_draft_smoke.sh data      # split manifest (needs the GSM8K cache; login node if missing)
#   bash tacc/selective_draft_smoke.sh backend   # inspect: module paths, token ids, trainable names
#   bash tacc/selective_draft_smoke.sh decode    # base collect on N_COLLECT analysis problems (threshold 0.9) + replay check
#   bash tacc/selective_draft_smoke.sh parity    # readout_only / zero-injection vs off on the same problems
#   bash tacc/selective_draft_smoke.sh readout   # stage 1, MAX_STEPS updates, then the readout report
#   bash tacc/selective_draft_smoke.sh teacher   # target-collapse + oracle-ceiling report on the collected states
#   bash tacc/selective_draft_smoke.sh guidance  # stage 2, MAX_STEPS updates with the stage-1 snapshot
#   bash tacc/selective_draft_smoke.sh resume    # stage 2 again from its checkpoint, +MAX_STEPS
#   bash tacc/selective_draft_smoke.sh eval      # guided vs base on N_EVAL dev problems, threshold sweep
#   bash tacc/selective_draft_smoke.sh consistency  # PSA / PMI / regret on the collected states
# Knobs: CONFIG, OUT, N_COLLECT (8), N_EVAL (8), MAX_STEPS (2), EARLY_LAYER (28), DEVICE.
set -euo pipefail
cd "$(dirname "$0")/.."
source tacc/env.sh

CONFIG="${CONFIG:-configs/selective_draft/llada_threshold.json}"
OUT="${OUT:-$SCRATCH/dopsd/selective_draft/smoke}"
N_COLLECT="${N_COLLECT:-8}"
N_EVAL="${N_EVAL:-8}"
MAX_STEPS="${MAX_STEPS:-2}"
EARLY_LAYER="${EARLY_LAYER:-28}"
MANIFEST="$OUT/data/split_manifest.json"
SET=(--set "guidance.early_layer=$EARLY_LAYER" --set "data.split_manifest=\"$MANIFEST\"")
mkdir -p "$OUT"
CLI="python -m selective_draft.cli"

stage_env() {
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
    python -c "import torch, transformers, safetensors; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
    python -c "import selective_draft, selective_draft.cli; print('selective_draft', selective_draft.__version__)"
    echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"
}
stage_data()     { $CLI prepare-data --config "$CONFIG" --output-dir "$OUT/data"; }
stage_backend()  { $CLI inspect --config "$CONFIG" "${SET[@]}" | head -80; }
stage_decode()   { $CLI collect --config "$CONFIG" "${SET[@]}" --set guidance.mode='"off"' --limit "$N_COLLECT" --output-dir "$OUT/base_states"; }
stage_parity() {
    $CLI evaluate --config "$CONFIG" "${SET[@]}" --set guidance.mode='"off"'          --set 'evaluation.commit_thresholds=[0.9]' --limit "$N_COLLECT" --output-dir "$OUT/parity_off"
    $CLI evaluate --config "$CONFIG" "${SET[@]}" --set guidance.mode='"readout_only"' --set 'evaluation.commit_thresholds=[0.9]' --limit "$N_COLLECT" --output-dir "$OUT/parity_readout"
    $CLI evaluate --config "$CONFIG" "${SET[@]}" --set guidance.mode='"single_soft"'  --set 'evaluation.commit_thresholds=[0.9]' --limit "$N_COLLECT" --output-dir "$OUT/parity_zero_inject"
    python - "$OUT" <<'PY'
import json, sys
o = sys.argv[1]
rows = {k: [json.loads(l) for l in open(f"{o}/{k}/rows_thr0.9.jsonl")] for k in ("parity_off", "parity_readout", "parity_zero_inject")}
for k in ("parity_readout", "parity_zero_inject"):
    same = sum(a["generation"] == b["generation"] for a, b in zip(rows["parity_off"], rows[k]))
    print(f"{k}: identical generations {same}/{len(rows[k])}; latency median off={json.load(open(f'{o}/parity_off/summary.json'))['summaries']['thr0.9']['latency_median_s']:.3f}s "
          f"{k}={json.load(open(f'{o}/{k}/summary.json'))['summaries']['thr0.9']['latency_median_s']:.3f}s")
PY
}
stage_readout() {
    $CLI train-readout --config "$CONFIG" "${SET[@]}" --states "$OUT/base_states" --max-steps "$MAX_STEPS" --output-dir "$OUT/readout"
    $CLI diagnose --kind readout --config "$CONFIG" "${SET[@]}" --snapshot "$OUT/readout/snapshot" --states "$OUT/base_states" --max-states 64 --output-dir "$OUT/readout_diag"
}
stage_teacher()  { $CLI diagnose --kind teacher --config "$CONFIG" "${SET[@]}" --snapshot "$OUT/readout/snapshot" --states "$OUT/base_states" --max-states 32 --output-dir "$OUT/teacher_diag"; }
stage_guidance() { $CLI train-guidance --config "$CONFIG" "${SET[@]}" --set data.split='"train"' --readout-snapshot "$OUT/readout/snapshot" --max-steps "$MAX_STEPS" --output-dir "$OUT/guidance"; }
stage_resume()   { $CLI train-guidance --config "$CONFIG" "${SET[@]}" --set data.split='"train"' --resume "$OUT/guidance/checkpoint-step-$MAX_STEPS" --max-steps $((2 * MAX_STEPS)) --output-dir "$OUT/guidance"; }
stage_eval() {
    $CLI evaluate --config "$CONFIG" "${SET[@]}" --set data.split='"dev"' --set guidance.mode='"off"' --limit "$N_EVAL" --output-dir "$OUT/eval_base"
    $CLI evaluate --config "$CONFIG" "${SET[@]}" --set data.split='"dev"' --set guidance.mode='"single_soft"' --snapshot "$OUT/guidance/snapshot" --limit "$N_EVAL" --output-dir "$OUT/eval_guided"
}
stage_consistency() { $CLI diagnose --kind consistency --config "$CONFIG" "${SET[@]}" --states "$OUT/base_states" --max-states 200 --output-dir "$OUT/consistency_base"; }

case "${1:-}" in
    env) stage_env ;; data) stage_data ;; backend) stage_backend ;; decode) stage_decode ;; parity) stage_parity ;;
    readout) stage_readout ;; teacher) stage_teacher ;; guidance) stage_guidance ;; resume) stage_resume ;;
    eval) stage_eval ;; consistency) stage_consistency ;;
    *) echo "usage: $0 {env|data|backend|decode|parity|readout|teacher|guidance|resume|eval|consistency}"; exit 1 ;;
esac
