#!/usr/bin/env bash
# Source this file inside the model container. Override values before sourcing if needed.
if mountpoint -q /root/private_data && [ -f "${HKS_ROOT:-/root/private_data/hks}/platform/scnet-notebook/env-scnet.sh" ]; then
    source "${HKS_ROOT:-/root/private_data/hks}/platform/scnet-notebook/env-scnet.sh"
    export PATH="/usr/bin:$PATH"
    export EXECUTION_PLATFORM="SCNet Notebook"
    if [ -f "$HKS_ROOT/weights/MiMo-V2.6-Distill-Qwen-9B/upstream-source.json" ]; then
        export MODEL_PATH="${MODEL_PATH:-$HKS_ROOT/weights/MiMo-V2.6-Distill-Qwen-9B}"
    else
        export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B}"
    fi
    export IMAGE="${IMAGE:-image.ac.com:5000/dcu/zack233/base/sglang-jupyter@sha256:98dca2a455d62b758a2455e674fb480bc39ac236a2b3b472de8a1525743ec8a3f}"
    export CARD="${CARD:-BW}"
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-results/scnet/runtime}"
    export SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PWD/results/scnet/smoke}"
fi
export MODEL_ID="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
export MODEL_PATH="${MODEL_PATH:-/data1/model/MiMo-V2.6-Distill-Qwen-9B}"
export SERVED_NAME="${SERVED_NAME:-mimo-v2.6-9b}"
export IMAGE="${IMAGE:-42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260928-0202}"
export CONTAINER_NAME="zk-MiMo-V2.6-Distill-Qwen-9B"
export FRAMEWORK="sglang"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
export CARD="${CARD:-BW100}"
export PRECISION="bf16"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
export HOST="127.0.0.1"
export PORT="${PORT:-30124}"
export TP="${TP:-1}"
export EP="${EP:-1}"
export PP="${PP:-1}"
export MAX_LEN="${MAX_LEN:-32768}"
export MEM_FRAC="${MEM_FRAC:-0.85}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
export DECODE_GRAPH_BACKEND="${DECODE_GRAPH_BACKEND:-full}"
export DECODE_GRAPH_MAX_BS="${DECODE_GRAPH_MAX_BS:-64}"
export DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-results}"
export LOAD_TIMEOUT="${LOAD_TIMEOUT:-1800}"
export EXTRA_ARGS="${EXTRA_ARGS:---reasoning-parser mimo}"
export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export MODELSCOPE_CACHE="$PROJECT_ROOT/cache/modelscope"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/cache/torchinductor}"
export PYTHONUNBUFFERED=1
export SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS=1
export SGLANG_KV_LAYOUT_HCU_FA=false
export TOKENIZERS_PARALLELISM=false
