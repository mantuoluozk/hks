#!/usr/bin/env bash
set -euo pipefail
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/bin:$PATH"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B}"
export IMAGE="${IMAGE:-image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18}"
export EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-}"
export EXECUTION_PLATFORM='SCNet Notebook'
export CARD=BW HIP_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/cache/huggingface"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
if [ -z "${ASR_EXTRA_DIR:-}" ]; then
  if [ -d /opt/qwen3-asr-deps ]; then ASR_EXTRA_DIR=/opt/qwen3-asr-deps; else ASR_EXTRA_DIR="$PWD/cache/python-deps"; fi
fi
export ASR_EXTRA_DIR
export PYTHONPATH="$ASR_EXTRA_DIR${PYTHONPATH:+:$PYTHONPATH}"
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/scnet/runtime}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [ "$#" -gt 0 ]; then exec "$@"; else exec bash run.sh; fi
