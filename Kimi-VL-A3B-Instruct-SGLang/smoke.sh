#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash run-scnet.sh python scripts/smoke_http.py --output "${RUN_RESULTS_DIR:-$PWD/results/functional-$(date -u +%Y%m%dT%H%M%SZ)}"
