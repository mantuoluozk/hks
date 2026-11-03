#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p results
RUN_RESULTS_DIR="$PWD/results/environment-check" python scripts/preflight.py > results/environment-check.log
cp results/environment-check/summary.json env.log
