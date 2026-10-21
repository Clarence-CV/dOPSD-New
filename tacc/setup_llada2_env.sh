#!/bin/bash
# One-time setup for LLaDA2.0-mini on a Lonestar6 LOGIN node (needs internet; no GPU work).
#   bash tacc/setup_llada2_env.sh            # env + downloads + offline check
#   bash tacc/setup_llada2_env.sh download   # downloads + offline check only
# The mini remote code targets Transformers 4.57.1 (config.json); the dOPSD env has 4.49, so a clone of it gets
# 4.57.1 as conda env `llada2`. Model and datasets are pinned to the revisions in configs/selective_draft/llada2_mini_math.json.
set -euo pipefail
cd "$(dirname "$0")/.."
ONLINE=1 source tacc/env.sh || true
MODEL=inclusionAI/LLaDA2.0-mini
MODEL_REV=dad945cac317da394b390f82c7b40691d8a881ed
source "$MINIFORGE/etc/profile.d/conda.sh"
if [[ "${1:-}" != "download" ]]; then
    conda env list | grep -q '^llada2 ' || conda create -y -n llada2 --clone dOPSD
    conda activate llada2
    pip install "transformers==4.57.1"
    python -c "import torch, transformers, huggingface_hub, datasets; print('torch', torch.__version__, '| transformers', \
transformers.__version__, '| hub', huggingface_hub.__version__, '| datasets', datasets.__version__)"
fi
conda activate llada2
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
for name, rev in (("ankner/math-500", "412eb03a7193aed402c5e4009fc958c66e61b937"),
                  ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")):
    ds = load_dataset(name, revision=rev)
    print("cached", name, rev[:8], {k: len(v) for k, v in ds.items()})
PY
# The exact offline path compute nodes take: config validation, split manifest, prompts.
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
from transformers import AutoTokenizer
from selective_draft.config import Config
from selective_draft.data import load_split
cfg = Config.load("configs/selective_draft/llada2_mini_math.json").validate()
tok = AutoTokenizer.from_pretrained(cfg.model.path, revision=cfg.model.revision, trust_remote_code=True)
for split, sub in (("test", None), ("dev", "dev_fast"), ("analysis", "readout_val"), ("train", "readout_train_initial")):
    cfg.data.split, cfg.data.subsplit = split, sub
    ex = load_split(cfg, tok)
    print("offline ok", split, sub, len(ex), "| prompt tokens", len(tok(ex[0].prompt_text)["input_ids"]))
print(repr(ex[0].prompt_text[:400]))
PY
echo "[setup_llada2_env] done. Quota check: /usr/local/etc/taccinfo"
