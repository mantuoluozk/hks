#!/usr/bin/env bash
set -euo pipefail
export BATCH_BENCH=0
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run.sh"
