#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if [ "${ASR_USE_RUNNING_SERVICE:-0}" = 1 ]; then
 exec python -u scripts/evaluate_librispeech.py --output "$RUN_RESULTS_DIR/accuracy"
fi
exec python -u scripts/job.py --stage accuracy
