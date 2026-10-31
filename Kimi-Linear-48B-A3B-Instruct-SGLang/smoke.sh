#!/usr/bin/env bash
# Functional smoke checks against an already running service (does not start it).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
OUT="${1:-$PWD/results/functional-reproduce}"
exec bash run-scnet.sh python scripts/http_checks.py --output "$OUT"
