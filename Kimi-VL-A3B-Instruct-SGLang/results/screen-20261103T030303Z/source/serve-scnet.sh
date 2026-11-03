#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash run-scnet.sh python scripts/launch.py \
  --model-path "$PWD/weights" --served-model-name kimi-vl-a3b-instruct \
  --host 127.0.0.1 --port 30004 --trust-remote-code \
  --dtype bfloat16 --tp-size 1 --context-length 24576 \
  --mem-fraction-static 0.80 --max-running-requests 8 \
  --attention-backend hcu_mla --moe-runner-backend triton \
  --mm-attention-backend sdpa --chunked-prefill-size 4096 \
  --disable-radix-cache --random-seed 42 "$@"
