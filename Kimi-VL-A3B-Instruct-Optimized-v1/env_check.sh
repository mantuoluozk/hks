#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
exec bash run-scnet.sh python scripts/preflight.py
