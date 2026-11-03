#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
OUT="${RUN_RESULTS_DIR:-$PWD/results/preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
export RUN_RESULTS_DIR="$OUT"
python -u scripts/preflight.py 2>&1 | tee "$OUT/env.log"
