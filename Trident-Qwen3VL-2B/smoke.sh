#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
out="${RUN_RESULTS_DIR:-$PWD/results/smoke}"
RUN_RESULTS_DIR="$out/publisher_reference" python scripts/reference_smoke.py
RUN_RESULTS_DIR="$out/native_embedding" REFERENCE_RESULTS="$out/publisher_reference/summary.json" python scripts/native_smoke.py
