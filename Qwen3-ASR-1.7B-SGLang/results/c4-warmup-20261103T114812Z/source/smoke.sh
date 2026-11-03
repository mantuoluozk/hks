#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
export SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PWD/results/smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
ln -sfn "${SMOKE_OUTPUT_DIR}.log" "$PWD/smoke.log"
python -u scripts/smoke.py 2>&1 | tee "${SMOKE_OUTPUT_DIR}.log"
