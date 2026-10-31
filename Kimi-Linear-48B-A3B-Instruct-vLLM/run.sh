#!/usr/bin/env bash
# Default launcher for Kimi-Linear-48B-A3B-Instruct on the SCNet four-card image.
# Starts the SGLang TP4 service used for every result in this directory.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_RESULTS_DIR"
if [[ "${CHECK_WEIGHTS_ON_START:-1}" == "1" ]]; then
  bash run-scnet.sh python scripts/check_weights.py "$PWD/weights" \
    --output "$RUN_RESULTS_DIR/weights-check.json"
fi
exec bash run-sglang.sh
