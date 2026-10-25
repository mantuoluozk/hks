#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs results
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/smoke_test.py 2>&1 | tee "logs/smoke-$STAMP.log"

