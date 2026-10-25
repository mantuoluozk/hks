#!/usr/bin/env bash
# Source this file inside the model container. Override values before sourcing if needed.
export MODEL_ID="BAAI/Recon2Reason-Reasoning-4B"
export MODEL_PATH="${MODEL_PATH:-/data1/model/Recon2Reason-Reasoning-4B}"
export SERVED_NAME="${SERVED_NAME:-recon2reason-4b}"
export IMAGE="${IMAGE:-42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260928-0202}"
export CONTAINER_NAME="zk-Recon2Reason-Reasoning-4B"
export FRAMEWORK="sglang"
export CARD="${CARD:-BW100}"
export PRECISION="bf16"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6}"
export HOST="127.0.0.1"
export PORT="${PORT:-30123}"
export TP="${TP:-1}"
export MAX_LEN="${MAX_LEN:-32768}"
export MEM_FRAC="${MEM_FRAC:-0.70}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
export DECODE_GRAPH_BACKEND="${DECODE_GRAPH_BACKEND:-full}"
export DECODE_GRAPH_MAX_BS="${DECODE_GRAPH_MAX_BS:-32}"
export LOAD_TIMEOUT="${LOAD_TIMEOUT:-1800}"
export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export MODELSCOPE_CACHE="$PROJECT_ROOT/cache/modelscope"
export TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/torchinductor"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

