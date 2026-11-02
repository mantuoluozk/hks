#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash ./run-scnet.sh python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH:-$PWD/weights}" \
  --served-model-name kimi-vl-a3b-instruct \
  --host 127.0.0.1 --port 30004 \
  --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 --enforce-eager \
  --limit-mm-per-prompt '{"image":2}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching --generation-config vllm \
  --seed 42 --no-enable-log-requests
