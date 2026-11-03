#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if [ "$#" -eq 0 ]; then exec python scripts/preflight.py; else exec "$@"; fi
