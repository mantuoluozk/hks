#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/bin:$PATH"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Qwen3-ForcedAligner-0.6B/main/Qwen3-ForcedAligner-0.6B}"
export HIP_VISIBLE_DEVICES=0
export EXECUTION_PLATFORM='SCNet Notebook' CARD=BW
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/cache/huggingface"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight}"
if [ "$#" -eq 0 ]; then exec python scripts/preflight.py; else exec "$@"; fi
