#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Starts and stops its own services; port 30003 must be free before running.
exec bash run-scnet.sh python scripts/run_service_validation.py
