#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/scnet/publisher-runtime-c64-20261026}"
export SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PWD/results/scnet/publisher-c64-smoke-20261026}"
backup_shared() {
  local script="${HKS_ROOT:-/root/private_data/hks}/platform/scnet-notebook/backup_shared.py"
  if [ -f "$script" ]; then python "$script"; fi
}
trap backup_shared EXIT
curl --noproxy '*' -fsS --max-time 4 http://127.0.0.1:30125/health >/dev/null
bash run-scnet.sh python scripts/prepare_cvbench.py
bash run-scnet.sh python scripts/smoke_test.py
bash run-scnet.sh python scripts/evaluate_cvbench.py \
  --runtime-dir "$RUN_RESULTS_DIR" \
  --output "${CVBENCH_OUTPUT_DIR:-results/scnet/publisher-cvbench-full-20261026}" \
  --workers "${CVBENCH_WORKERS:-2}"
