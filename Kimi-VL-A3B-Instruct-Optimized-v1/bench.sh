#!/usr/bin/env bash
# Run one official-template matrix; use --processes 2 for independent-service reproduction.
# Port 30004 must be free. Each cell has two repetitions; the second is measured.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash run-scnet.sh python -u scripts/run_formal_matrix.py "$@"
