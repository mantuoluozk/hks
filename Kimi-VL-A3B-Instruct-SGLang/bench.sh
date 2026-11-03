#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Standalone full validation: own service, OCRBench, context probe, 608-request matrix, stop.
exec bash run-scnet.sh python scripts/run_formal_matrix.py --processes 1
