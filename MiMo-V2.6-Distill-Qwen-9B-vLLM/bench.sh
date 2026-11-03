#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
python -u scripts/formal.py
python -u scripts/audit_formal.py
