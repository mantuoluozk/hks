#!/usr/bin/env bash
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="${MODEL_PATH:-$HKS_ROOT/weights/MiMo-V2.6-Distill-Qwen-9B}"
export SERVED_NAME=mimo-v26-vllm PORT=30128
export MAX_LEN="${MAX_LEN:-32768}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" MEM_FRAC="${MEM_FRAC:-0.80}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HIP_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg" TRITON_CACHE_DIR="$PWD/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export EXECUTION_MODE="${EXECUTION_MODE:-graph}"
