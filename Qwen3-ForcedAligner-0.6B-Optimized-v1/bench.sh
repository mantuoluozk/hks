#!/usr/bin/env bash
set -euo pipefail
export BATCH_BENCH=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run.sh"
