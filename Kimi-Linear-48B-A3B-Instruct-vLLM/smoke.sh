#!/usr/bin/env bash
# Functional smoke checks against an already running vLLM service.
# Does NOT start or stop the service.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
OUT="${1:-$PWD/results/functional-reproduce}"
exec bash run-scnet.sh python scripts/vllm_smoke.py --output "$OUT"
