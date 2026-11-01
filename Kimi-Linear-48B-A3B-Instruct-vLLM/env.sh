#!/usr/bin/env bash
# Runtime configuration for the Kimi-Linear vLLM TP4 service.
# Defaults match the validated configuration used in README sections 2 and 7
# (TP4, four BW cards, port 30149). Export any of these to override.
export MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/weights}"
export MODEL_NAME="${MODEL_NAME:-kimi-linear-48b-a3b}"

export PORT="${PORT:-30149}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"

if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  case "$TENSOR_PARALLEL_SIZE" in
    4) export HIP_VISIBLE_DEVICES="0,1,2,3" ;;
    2) export HIP_VISIBLE_DEVICES="0,1" ;;
    *) export HIP_VISIBLE_DEVICES="0" ;;
  esac
fi

# Expert parallelism and custom ops were disabled for every result here.
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
export VLLM_HCU_USE_CUSTOM_OPS="${VLLM_HCU_USE_CUSTOM_OPS:-0}"
export VLLM_KIMI_CONV1D_BACKEND="${VLLM_KIMI_CONV1D_BACKEND:-0}"
