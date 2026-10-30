#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
exec bash run-scnet.sh python scripts/http_checks.py smoke --base "http://127.0.0.1:$PORT" --model "$MODEL_NAME" --output "${1:-results/smoke}"
