#!/usr/bin/env bash
# One-model first-release validation in the official Notebook.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source env.sh
test "${EXECUTION_PLATFORM:-}" = 'SCNet Notebook'
mkdir -p "$RUN_RESULTS_DIR" results/scnet
phase() {
    export SCNET_PHASE="$1"
    python - <<'PY'
import datetime,json,os
from pathlib import Path
Path('results/scnet/progress.json').write_text(json.dumps({
    'phase':os.environ['SCNET_PHASE'],
    'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'execution_platform':'SCNet Notebook','official_acceptance_passed':None},indent=2)+'\n')
PY
}
phase kernel-check
python scripts/apply_deterministic_lds_patch.py
python scripts/check_deterministic_gemm.py > results/scnet/kernel-check.log 2>&1
phase server-start
bash run.sh > results/scnet/launch.log 2>&1 &
LAUNCHER=$!
READY=0
for ((i=0;i<360;i++)); do
    kill -0 "$LAUNCHER" 2>/dev/null || { wait "$LAUNCHER"; exit 1; }
    if curl --noproxy '*' -fsS --max-time 4 "http://$HOST:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
    sleep 5
done
test "$READY" = 1
phase functional
export SMOKE_OUTPUT_DIR="$PROJECT_ROOT/results/scnet/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/smoke_test.py > results/scnet/smoke.log 2>&1
phase performance
export BENCH_PROFILE=matrix
export BENCH_OUTPUT_DIR="bench/scnet-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$BENCH_OUTPUT_DIR" > results/scnet/benchmark-path.txt
bash bench.sh > results/scnet/benchmark.log 2>&1
phase first-validation-complete
python "$HKS_ROOT/platform/scnet-notebook/backup_shared.py" > results/scnet/backup.log 2>&1
