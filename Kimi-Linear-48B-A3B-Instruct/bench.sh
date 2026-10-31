#!/usr/bin/env bash
# Performance matrix used in README section 7: 2 prompt lengths x concurrency 1/2/4.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
OUT="${1:-$PWD/results/performance-reproduce}"
exec bash run-scnet.sh python scripts/bench_http.py --output "$OUT"
