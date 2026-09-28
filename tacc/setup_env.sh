#!/bin/bash
# One-time setup on a Lonestar6 LOGIN node (needs internet; no GPU work here).
#   bash tacc/setup_env.sh           # env + trl patch + downloads
#   bash tacc/setup_env.sh download  # downloads only
# Versions follow used-env.txt ("the real environment we used for all experiments");
# env.yml differs only in torch (2.6.0 there vs 2.9.0+cu128 here).
set -euo pipefail
cd "$(dirname "$0")/.."
ONLINE=1 source tacc/env.sh || true

TRL_COMMIT=0f88c179e30b3439467942a08c3190f624d5c423
MINIFORGE="$DOPSD_ROOT/miniforge3"
if [[ "${1:-}" != "download" ]]; then
    if [[ ! -d "$MINIFORGE" ]]; then
        curl -fsSL -o "/tmp/miniforge_$USER.sh" \
            https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash "/tmp/miniforge_$USER.sh" -b -p "$MINIFORGE"
        rm -f "/tmp/miniforge_$USER.sh"
    fi
    source "$MINIFORGE/etc/profile.d/conda.sh"
    conda env list | grep -q '^dOPSD ' || conda create -y -n dOPSD python=3.10
    conda activate dOPSD

    echo "torch==2.9.0" > "/tmp/dopsd_constraints_$USER.txt"
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -c "/tmp/dopsd_constraints_$USER.txt" \
        numpy==1.25.0 transformers==4.49.0 accelerate==1.4.0 bitsandbytes==0.45.3 peft==0.15.1 \
        "trl @ git+https://github.com/huggingface/trl.git@$TRL_COMMIT" \
        deepspeed==0.16.4 datasets==3.3.2 tiktoken==0.9.0 wandb==0.15.3 \
        sentencepiece evaluate scipy tqdm regex scikit-learn pandas \
        setuptools==69.5.1  # wandb 0.15.3 imports pkg_resources (removed in recent setuptools)
    rm -f "/tmp/dopsd_constraints_$USER.txt"

    # README step 2/3: the two edits d-OPSD needs in trl (num_generations=1 allowed;
    # teacher-prompt column in print_prompt_completions_sample). Applied as a patch
    # against the pinned commit so it fails loudly if the files differ.
    SP=$(python -c "import os, trl; print(os.path.dirname(os.path.dirname(trl.__file__)))")
    if grep -q "range(1, global_batch_size + 1)" "$SP/trl/trainer/grpo_trainer.py"; then
        echo "[setup_env] trl already patched"
    else
        patch -p1 -d "$SP" < tacc/trl-0f88c17-dopsd.patch
    fi
    python -c "import torch, transformers, trl, peft, deepspeed, bitsandbytes; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, \
'| trl', trl.__version__, '| peft', peft.__version__, '| deepspeed', deepspeed.__version__, '| bnb', bitsandbytes.__version__)"
fi

source "$MINIFORGE/etc/profile.d/conda.sh"; conda activate dOPSD
# Login nodes cap virtual memory at 8 GB (ulimit -v): download weights in a process that
# imports only huggingface_hub, with the Rust xet backend off and few workers.
HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import snapshot_download
print("cached", snapshot_download("GSAI-ML/LLaDA-8B-Instruct", max_workers=4))
PY
python - <<'PY'
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

m = "GSAI-ML/LLaDA-8B-Instruct"
# Populate the trust_remote_code module cache so offline compute nodes can import it.
AutoConfig.from_pretrained(m, trust_remote_code=True)
AutoTokenizer.from_pretrained(m, trust_remote_code=True)
# Training (d-opsd/data_utils.py) and eval (eval/*.py) datasets; "gsm8k" is the
# legacy name eval/gsm8k.py uses, cached separately from "openai/gsm8k".
for args in [("openai/gsm8k", "main"), ("gsm8k", "main"), ("ankner/math-500",),
             ("HuggingFaceH4/MATH-500",), ("Jiayi-Pan/Countdown-Tasks-3to4",)]:
    ds = load_dataset(*args)
    print("cached", args, {k: len(v) for k, v in ds.items()})
PY
echo "[setup_env] done. Quota check: /usr/local/etc/taccinfo"
