#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${RUN_RESULTS_DIR:?Use run-scnet.sh to configure this run}"
output="$RUN_RESULTS_DIR"
mkdir -p "$output"
RUN_RESULTS_DIR="$output/runtime" python scripts/check_weights.py > "$output/weights.log"
RUN_RESULTS_DIR="$output/preflight" python scripts/preflight.py > "$output/preflight.log"
RUN_RESULTS_DIR="$output/smoke" python scripts/smoke_offline.py > "$output/offline_alignment.log" 2>&1
