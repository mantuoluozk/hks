#!/usr/bin/env bash
# Stop a Kimi-Linear service started from this directory.
set -euo pipefail
PIDS=$(pgrep -f "sglang.launch_server.*kimi-linear-48b-a3b" || true)
if [[ -z "${PIDS}" ]]; then
  echo "no Kimi-Linear sglang.launch_server process found"
  exit 0
fi
echo "stopping: ${PIDS}"
kill ${PIDS}
for _ in $(seq 1 30); do
  pgrep -f "sglang.launch_server.*kimi-linear-48b-a3b" >/dev/null || { echo stopped; exit 0; }
  sleep 2
done
echo "force killing"
kill -9 ${PIDS} 2>/dev/null || true
