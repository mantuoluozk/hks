#!/usr/bin/env bash
# Kimi-Linear on the SCNet four-card image.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
source ./env.sh
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
exec bash run-scnet.sh python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" --tp "$TENSOR_PARALLEL_SIZE" \
  --dtype bfloat16 --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.80}" \
  --max-running-requests "$MAX_NUM_SEQS" \
  --disable-cuda-graph --attention-backend triton --trust-remote-code
