#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash serve-scnet.sh --cuda-graph-bs-decode 1 2 4 8 --enable-metrics
