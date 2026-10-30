#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
# The shared default was created for single-card models; this model uses both cards.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
export PATH="/usr/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HKS_ROOT/models/MiniCPM-V-4.7-35B-A3B/cache/huggingface"
export XDG_CACHE_HOME="$HKS_ROOT/models/MiniCPM-V-4.7-35B-A3B/cache/xdg"
export TRITON_CACHE_DIR="$HKS_ROOT/models/MiniCPM-V-4.7-35B-A3B/cache/triton"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/cache/runtime:$PWD/python:/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$@"
