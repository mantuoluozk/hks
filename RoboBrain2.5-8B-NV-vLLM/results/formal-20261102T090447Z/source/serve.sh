#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p results
python scripts/check_weights.py
exec python -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192 --gpu-memory-utilization "$MEM_FRAC" --compilation-config "$COMPILATION_CONFIG" --no-enable-prefix-caching --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":2,"video":0}'
