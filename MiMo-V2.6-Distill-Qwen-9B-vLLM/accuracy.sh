#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
python scripts/prepare_gsm8k.py
exec python -u scripts/accuracy.py
