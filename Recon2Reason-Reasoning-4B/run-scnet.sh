#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
# Use the same global interpreter as the validated SSH launch, including in IDE terminals.
export PATH="/usr/bin:$PATH"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B}"
export IMAGE="image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18"
export CARD="BW"
export EXECUTION_PLATFORM="SCNet Notebook"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ "$#" -gt 0 ]; then exec "$@"; else exec bash ./run.sh; fi
