#!/usr/bin/env bash
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
mountpoint -q /root/private_data || { echo 'Shared storage is not mounted' >&2; return 1; }
if [ -f "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh" ]; then
 export HKS_ENV_SOURCE=shared-platform
 source "$HKS_ROOT/platform/scnet-notebook/env-scnet.sh"
elif [ -f /opt/kimi-vl-vllm/platform/env-scnet.sh ]; then
 export HKS_ENV_SOURCE=image-platform
 source /opt/kimi-vl-vllm/platform/env-scnet.sh
else
 export HKS_ENV_SOURCE=image-dtk
 _asr_had_nounset=0
 case "$-" in *u*) _asr_had_nounset=1; set +u ;; esac
 source /opt/dtk/env.sh
 source /opt/hyhal/env.sh
 if [ "$_asr_had_nounset" = 1 ]; then set -u; fi
 unset _asr_had_nounset
fi
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Qwen3-ASR-1.7B/main/Qwen3-ASR-1.7B}"
export OLD_ASR_ROOT="${OLD_ASR_ROOT:-$HKS_ROOT/models/Qwen3-ASR-1.7B}"
export SERVED_NAME=qwen3-asr-sglang PORT=30129
export HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME="$PWD/cache/huggingface" XDG_CACHE_HOME="$PWD/cache/xdg" TRITON_CACHE_DIR="$PWD/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export EXECUTION_MODE="${EXECUTION_MODE:-graph}"
