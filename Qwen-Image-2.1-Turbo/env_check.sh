#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/environment-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_RESULTS_DIR"
bash run-scnet.sh python scripts/preflight.py 2>&1 | tee "$RUN_RESULTS_DIR/env.log"
