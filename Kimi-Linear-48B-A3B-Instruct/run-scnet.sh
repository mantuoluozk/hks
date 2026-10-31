#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
export PATH="/usr/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"
# Isolated runtime: the image ships openai 2.6.1, which lacks NamespaceTool required by
# this vLLM build's tool-parser import chain (first load failed on ImportError).
# Only openai is isolated; torch / vLLM / transformers stay at image versions.
export PYTHONPATH="${KIMI_LINEAR_RUNTIME:-$PWD/cache/runtime}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$@"
