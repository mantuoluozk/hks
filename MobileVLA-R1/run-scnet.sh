#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/bin:$PATH"
if [[ -d /opt/qwen3-asr-deps ]]; then
  export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$@"
