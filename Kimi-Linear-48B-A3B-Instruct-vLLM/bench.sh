#!/usr/bin/env bash
# Performance matrix used in README section 7: 2 prompt lengths x concurrency 1/2/4.
# Runs against an already running service; does not start it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
OUT="${1:-$PWD/results/performance-reproduce}"
exec bash run-scnet.sh python scripts/bench_vllm.py --output "$OUT" \
  --base "http://127.0.0.1:${PORT}" --model "$MODEL_NAME"
