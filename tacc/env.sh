# Common environment for every d-OPSD process on Lonestar6 (login node, idev, sbatch).
#   source tacc/env.sh            # compute node: offline HF/W&B
#   ONLINE=1 source tacc/env.sh   # login node: downloads allowed
# All caches are redirected off $HOME (10 GB quota).

export DOPSD_ROOT="${DOPSD_ROOT:-$WORK/inf385t}"
export HF_HOME="$DOPSD_ROOT/hf"
export PIP_CACHE_DIR="$DOPSD_ROOT/pip"
export TORCH_EXTENSIONS_DIR="$DOPSD_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$DOPSD_ROOT/triton"
export XDG_CACHE_HOME="$DOPSD_ROOT/cache"
export WANDB_DIR="$SCRATCH/dopsd"
export TOKENIZERS_PARALLELISM=false
# bitsandbytes 0.45.3 picks libbitsandbytes_cuda128.so for torch cu128, but that build (and
# cuda126) needs GLIBC 2.34; LS6 has 2.28. The cuda125 build only needs GLIBC 2.4 and runs
# against torch's CUDA 12.8 runtime (same libcudart.so.12 major).
export BNB_CUDA_VERSION=125
# `import deepspeed` on a GPU node checks the CUDA toolkit (CUDA_HOME) for its op builders.
module load cuda/12.8 >/dev/null 2>&1 || echo "[env.sh] WARN: could not load cuda/12.8"
export CUDA_HOME="${CUDA_HOME:-${TACC_CUDA_DIR:-}}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

export WANDB_MODE=offline   # sync later from a login node: wandb sync $WANDB_DIR/wandb/offline-run-*
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
    conda activate dOPSD
else
    echo "[env.sh] WARN: $MINIFORGE missing; run tacc/setup_env.sh on a login node first."
fi
