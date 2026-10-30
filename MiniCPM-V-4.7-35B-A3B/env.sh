#!/usr/bin/env bash
export MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/weights}"
export MODEL_NAME="${MODEL_NAME:-minicpm-v-4.7}"
export PORT="${PORT:-30147}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
