#!/usr/bin/env bash
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
if [ -f /opt/recon2reason-vllm/platform/env-scnet.sh ]; then
 source /opt/recon2reason-vllm/platform/env-scnet.sh
else
 source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
fi
export PATH="/usr/bin:$PATH"
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
export SERVED_NAME=robobrain25-8b-vllm
export PORT="${PORT:-30126}"
export MAX_LEN="${MAX_LEN:-32768}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MEM_FRAC="${MEM_FRAC:-0.80}" HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EXECUTION_PLATFORM='SCNet Notebook'
export TRITON_CACHE_DIR="$PWD/cache/triton" HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg"
if [ -z "${COMPILATION_CONFIG:-}" ]; then
 export COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
fi
