#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/bin:$PATH"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Qwen-Image-2.1-Turbo/main/Qwen-Image-2.1-Turbo}"
export HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HKS_IMAGE_RUNTIME="${HKS_IMAGE_RUNTIME:-/opt/qwen-image21}"
if [[ -f "$HKS_IMAGE_RUNTIME/site-packages/transformers/__init__.py" && -f "$HKS_IMAGE_RUNTIME/diffusers-py310/src/diffusers/__init__.py" ]]; then
  export HKS_RUNTIME_ROOT="$HKS_IMAGE_RUNTIME"
else
  export HKS_RUNTIME_ROOT="$PWD/runtime"
fi
export PYTHONPATH="$HKS_RUNTIME_ROOT/site-packages:$HKS_RUNTIME_ROOT/diffusers-py310/src:/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg"
export TRITON_CACHE_DIR="$PWD/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight}"
exec "$@"
