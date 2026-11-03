#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_RESULTS_DIR"
ln -sfn "$RUN_RESULTS_DIR/env.log" "$PWD/env.log"
python -u scripts/preflight.py 2>&1 | tee "$RUN_RESULTS_DIR/env.log"
