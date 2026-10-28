#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -c 'from scripts.mmlu_pro_common import assert_auk_idle; assert_auk_idle()'
export MAX_LEN=49152
export MAX_RUNNING_REQUESTS=1
export DECODE_GRAPH_MAX_BS=1
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/mmlu-pro/runtime}"
exec bash ./run-scnet.sh
