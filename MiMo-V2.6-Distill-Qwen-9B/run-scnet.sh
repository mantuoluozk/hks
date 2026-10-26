#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
# Use the same global interpreter as the validated SSH launch, including in IDE terminals.
export PATH="/usr/bin:$PATH"
if [ -f "$HKS_ROOT/weights/MiMo-V2.6-Distill-Qwen-9B/upstream-source.json" ]; then
    export MODEL_PATH="${MODEL_PATH:-$HKS_ROOT/weights/MiMo-V2.6-Distill-Qwen-9B}"
else
    export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B}"
fi
export IMAGE="image.ac.com:5000/dcu/zack233/base/sglang-jupyter@sha256:98dca2a455d62b758a2455e674fb480bc39ac236a2b3b472de8a1525743ec8a3f"
export CARD="BW"
export EXECUTION_PLATFORM="SCNet Notebook"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ "$#" -gt 0 ]; then exec "$@"; else exec bash ./run.sh; fi
