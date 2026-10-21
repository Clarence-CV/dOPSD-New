# Common environment for every process on Lonestar6 (login node, idev, sbatch).
#   source tacc/env.sh            # compute node: offline HF/W&B
#   ONLINE=1 source tacc/env.sh   # login node: downloads allowed
#   CONDA_ENV=other source tacc/env.sh   # another conda env (default: llada2, from tacc/setup_llada2_env.sh)
# All caches are redirected off $HOME (10 GB quota).

export DOPSD_ROOT="${DOPSD_ROOT:-$WORK/inf385t}"
export HF_HOME="$DOPSD_ROOT/hf"
export PIP_CACHE_DIR="$DOPSD_ROOT/pip"
export TORCH_EXTENSIONS_DIR="$DOPSD_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$DOPSD_ROOT/triton"
export XDG_CACHE_HOME="$DOPSD_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # project convention; nothing here logs to W&B
# CUDA toolkit module: torch wheels bring their own CUDA runtime, but the jobs were validated with this module
# loaded, so it stays.
module load cuda/12.8 >/dev/null 2>&1 || echo "[env.sh] WARN: could not load cuda/12.8"
export CUDA_HOME="${CUDA_HOME:-${TACC_CUDA_DIR:-}}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

if [[ "${ONLINE:-0}" == "1" ]]; then
    unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
else
    # Compute nodes have no internet: force local caches.
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

MINIFORGE="$DOPSD_ROOT/miniforge3"
if [[ -f "$MINIFORGE/etc/profile.d/conda.sh" ]]; then
    source "$MINIFORGE/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-llada2}"
else
    echo "[env.sh] WARN: $MINIFORGE missing; run tacc/setup_llada2_env.sh on a login node first."
fi
