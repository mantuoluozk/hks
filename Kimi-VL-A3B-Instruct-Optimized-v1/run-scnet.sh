#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
if [[ -f "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh" ]]; then
  source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
else
  source /opt/kimi-vl-vllm/platform/env-scnet.sh
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Reuse the fixed API dependencies already bundled in the ASR/Qwen-Image image.
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="${MODEL_PATH:-$PWD/weights}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# The fused HCU gate excludes Kimi-VL's 64-expert/top-6 shape. Use its standard router.
export VLLM_HCU_USE_FUSE_MOE_GATE=0
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight-20261028}"
mkdir -p "$RUN_RESULTS_DIR"
exec "$@"
