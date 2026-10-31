#!/usr/bin/env bash
export MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/weights}"
export MODEL_NAME="${MODEL_NAME:-kimi-linear-48b-a3b}"
export PORT="${PORT:-30148}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$TENSOR_PARALLEL_SIZE" == "4" ]]; then
    export HIP_VISIBLE_DEVICES="0,1,2,3"
  else
    export HIP_VISIBLE_DEVICES="0,1"
  fi
fi
