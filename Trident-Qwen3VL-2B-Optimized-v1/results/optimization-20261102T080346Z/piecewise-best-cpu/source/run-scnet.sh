#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/bin:$PATH"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Trident-Qwen3VL-2B/main/Trident-Qwen3VL-2B}"
export HIP_VISIBLE_DEVICES=0 EXECUTION_PLATFORM='SCNet Notebook' CARD=BW
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/cache/huggingface" TRITON_CACHE_DIR="$PWD/cache/triton"
export XDG_CACHE_HOME="$PWD/cache/xdg" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export PYTHONPATH="$PWD/sources/vendor:$PWD/sources/upstream/src:/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${TRIDENT_CPU_THREADS:-4}" OPENBLAS_NUM_THREADS="${TRIDENT_CPU_THREADS:-4}" MKL_NUM_THREADS="${TRIDENT_CPU_THREADS:-4}" NUMEXPR_NUM_THREADS="${TRIDENT_CPU_THREADS:-4}"
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight}"
if [ "$#" -eq 0 ]; then exec python scripts/preflight.py; else exec "$@"; fi
