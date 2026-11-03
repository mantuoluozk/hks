#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if [ "${ASR_USE_RUNNING_SERVICE:-0}" = 1 ]; then
 python -u scripts/bench_audio.py --output "$RUN_RESULTS_DIR/bench"
 exec python -u scripts/audit_audio.py --folder "$RUN_RESULTS_DIR/bench" --publish-index
fi
exec python -u scripts/job.py --stage bench
