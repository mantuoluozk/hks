#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python scripts/check_weights.py
python scripts/preflight.py
bash smoke.sh
