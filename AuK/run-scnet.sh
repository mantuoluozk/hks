#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${HKS_AUK_RUNTIME:-}" && -d /opt/auk/sglang-omni/sglang_omni ]]; then
  export HKS_AUK_RUNTIME=/opt/auk
fi
if [[ -n "${HKS_AUK_RUNTIME:-}" ]]; then
  test -d "$HKS_AUK_RUNTIME/sglang-omni/sglang_omni"
  export PYTHONPATH="$HKS_AUK_RUNTIME/site-packages:$HKS_AUK_RUNTIME/sglang-omni:/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$PWD/runtime/deps:$PWD/runtime/sglang-omni:/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
fi
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/runtime-preflight-20261028}"
mkdir -p "$RUN_RESULTS_DIR"
exec "$@"
