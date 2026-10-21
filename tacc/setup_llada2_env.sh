#!/bin/bash
# One-time setup for LLaDA2.0-mini on a Lonestar6 LOGIN node (needs internet; no GPU work).
#   bash tacc/setup_llada2_env.sh                         # miniforge + env + downloads + offline check
#   bash tacc/setup_llada2_env.sh download                # downloads + offline check only
#   ENV_NAME=llada2_check bash tacc/setup_llada2_env.sh env   # only (re)create an env, e.g. to test the recipe
# Model and datasets are pinned to the revisions in configs/llada2_mini_math.json. Python packages: requirements-llada2.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_ENV=base ONLINE=1 source tacc/env.sh || true
ENV_NAME="${ENV_NAME:-llada2}"
MODEL=inclusionAI/LLaDA2.0-mini
MODEL_REV=dad945cac317da394b390f82c7b40691d8a881ed
MINIFORGE="$DOPSD_ROOT/miniforge3"
if [[ ! -f "$MINIFORGE/etc/profile.d/conda.sh" ]]; then
    curl -fsSL -o "/tmp/miniforge_$USER.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "/tmp/miniforge_$USER.sh" -b -p "$MINIFORGE"
    rm -f "/tmp/miniforge_$USER.sh"
fi
source "$MINIFORGE/etc/profile.d/conda.sh"
if [[ "${1:-}" != "download" ]]; then
    conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
    conda activate "$ENV_NAME"
    pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements-llada2.txt
    python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('torch', 'transformers', 'datasets', 'numpy', 'math-verify', 'sympy')})"
fi
[[ "${1:-}" == "env" ]] && { echo "[setup_llada2_env] env $ENV_NAME ready"; exit 0; }
conda activate "$ENV_NAME"
# Login nodes cap virtual memory at 8 GB: download in a process that imports only huggingface_hub, xet off.
HF_HUB_DISABLE_XET=1 python - "$MODEL" "$MODEL_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("cached", snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=4))
PY
python - "$MODEL" "$MODEL_REV" <<'PY'
import sys
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
m, rev = sys.argv[1], sys.argv[2]
# Populates the trust_remote_code module cache for offline compute nodes.
cfg = AutoConfig.from_pretrained(m, revision=rev, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(m, revision=rev, trust_remote_code=True)
assert tok.convert_tokens_to_ids("<|mask|>") == 156895 and tok.eos_token_id == 156892, "special ids changed"
print("config", cfg.model_type, cfg.num_hidden_layers, "| mask", tok.convert_tokens_to_ids("<|mask|>"))
for name, r in (("ankner/math-500", "412eb03a7193aed402c5e4009fc958c66e61b937"),
                ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")):
    ds = load_dataset(name, revision=r)
    print("cached", name, r[:8], {k: len(v) for k, v in ds.items()})
PY
# The exact offline path compute nodes take: config validation, split manifest, prompts.
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
from transformers import AutoTokenizer
from selective_draft.config import Config
from selective_draft.data import load_split
cfg = Config.load("configs/llada2_mini_math.json").validate()
tok = AutoTokenizer.from_pretrained(cfg.model.path, revision=cfg.model.revision, trust_remote_code=True)
for split, sub in (("test", None), ("dev", "dev_fast"), ("analysis", "readout_val"), ("train", "readout_train_initial")):
    cfg.data.split, cfg.data.subsplit = split, sub
    ex = load_split(cfg, tok)
    print("offline ok", split, sub, len(ex), "| prompt tokens", len(tok(ex[0].prompt_text)["input_ids"]))
print(repr(ex[0].prompt_text[:400]))
PY
echo "[setup_llada2_env] done. Quota check: /usr/local/etc/taccinfo"
