#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
OUT="${SMOKE_OUTPUT_DIR:-$PWD/results/smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
python -u scripts/smoke.py --output "$OUT" 2>&1 | tee "${OUT}.log"
