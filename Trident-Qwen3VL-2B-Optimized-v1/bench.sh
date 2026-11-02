#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
out="${RUN_RESULTS_DIR:-$PWD/results/benchmark}"
if [ -z "${REFERENCE_RESULTS:-}" ]; then
  RUN_RESULTS_DIR="$out/publisher_reference" python scripts/reference_smoke.py
  export REFERENCE_RESULTS="$out/publisher_reference/summary.json"
fi
RUN_RESULTS_DIR="$out/performance" python scripts/benchmark_native.py
