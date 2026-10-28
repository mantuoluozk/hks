#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
smoke="results/scnet/native-vllm-v3-20261028/functional/summary.json"
bash run-scnet.sh python scripts/eval_ocrbench.py --smoke-summary "$smoke" \
  --output results/scnet/ocrbench-pilot-20261028 --limit 8
exec bash run-scnet.sh python scripts/eval_ocrbench.py --smoke-summary "$smoke" \
  --output results/scnet/ocrbench-full-20261028
