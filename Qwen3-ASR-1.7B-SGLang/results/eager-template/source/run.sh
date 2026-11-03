#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/runtime-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_RESULTS_DIR"
python scripts/prepare_runtime.py
ln -sfn "$RUN_RESULTS_DIR/server.log" "$PWD/server.log"
ARGS=(--model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$PORT" --chat-template "$PWD/runtime/qwen3-asr.jinja"
 --dtype bfloat16 --tp-size 1 --context-length 8192 --mem-fraction-static 0.60 --max-running-requests 32
 --attention-backend triton --mm-attention-backend sdpa --chunked-prefill-size -1
 --disable-radix-cache --mm-preprocess-cache-size-mb 0 --random-seed 42 --enable-metrics)
if [ "$EXECUTION_MODE" = eager ]; then ARGS+=(--disable-cuda-graph)
elif [ "$EXECUTION_MODE" = graph ]; then ARGS+=(--cuda-graph-bs-decode 1 2 4 8 16)
else echo 'EXECUTION_MODE must be eager or graph' >&2; exit 1; fi
exec python -u scripts/service.py "${ARGS[@]}"
