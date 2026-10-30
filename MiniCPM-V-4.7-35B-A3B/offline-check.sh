#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bash run-scnet.sh python scripts/check_runtime.py
bash run-scnet.sh python -m pytest python/minicpm_adapter/test_upstream.py python/minicpm_adapter/test_parser.py -q
for f in ./*.sh; do bash -n "$f"; done
