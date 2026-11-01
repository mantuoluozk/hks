#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs
python scripts/smoke_test.py | tee "logs/smoke-$(date -u +%Y%m%dT%H%M%SZ).log"
