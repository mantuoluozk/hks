#!/usr/bin/env bash
# SCNet runtime wrapper for the Kimi-Linear vLLM service on the four-card BW image.
#
# The image ships openai 2.6.1, which lacks NamespaceTool required by this vLLM
# build's tool-parser import chain. Only `openai` is isolated into a local
# runtime directory; torch / vLLM / transformers / boltops stay at image versions.
#
# A fresh container has no cache/runtime yet. `scripts/prepare_runtime.py`
# builds it (pip download/install of openai 2.30.0) so a new Notebook can be
# reproduced with one command instead of depending on a pre-existing cache.
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh" 2>/dev/null || true
export PATH="/usr/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"

# Isolated openai runtime. Point KIMI_LINEAR_RUNTIME at a prepared directory, or
# let it default to cache/runtime and build it on first use.
KIMI_LINEAR_RUNTIME="${KIMI_LINEAR_RUNTIME:-$PWD/cache/runtime}"
if [[ ! -d "$KIMI_LINEAR_RUNTIME/openai" ]]; then
  if [[ "${KIMI_AUTO_PREPARE_RUNTIME:-1}" == "1" ]]; then
    echo "[run-scnet] isolated openai runtime missing; preparing into $KIMI_LINEAR_RUNTIME" >&2
    python scripts/prepare_runtime.py --dest "$KIMI_LINEAR_RUNTIME" || {
      echo "[run-scnet] WARNING: runtime preparation failed. If the image openai is" >&2
      echo "             already new enough, this may still work; otherwise run" >&2
      echo "             scripts/prepare_runtime.py manually with network access." >&2
    }
  fi
fi
export PYTHONPATH="$KIMI_LINEAR_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$@"
