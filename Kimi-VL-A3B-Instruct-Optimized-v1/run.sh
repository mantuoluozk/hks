#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
case "$HKS_PROFILE" in
  baseline) launcher=serve-scnet.sh ;;
  short-optimized) launcher=serve-short-optimized.sh ;;
  optimized) launcher=serve-optimized.sh ;;
  *) echo "Unknown HKS_PROFILE: $HKS_PROFILE" >&2; exit 2 ;;
esac
python - <<'PY'
import socket
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',30004)) == 0:
        raise SystemExit('Port 30004 already has a listener; leave the existing service intact')
PY
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
bash run-scnet.sh python scripts/preflight.py
printf '%s\n' "$$" > "$PWD/results/service.pid"
exec bash "$launcher"
