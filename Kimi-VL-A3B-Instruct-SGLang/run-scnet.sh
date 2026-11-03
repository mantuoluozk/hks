#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg"
export TRITON_CACHE_DIR="$PWD/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export OMP_NUM_THREADS="${CPU_THREADS:-4}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-4}" MKL_NUM_THREADS="${CPU_THREADS:-4}"
exec "$@"
