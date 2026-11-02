#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# This controller owns one service and refuses to run if its port is occupied.
exec bash run-scnet.sh python scripts/formal_matrix.py
