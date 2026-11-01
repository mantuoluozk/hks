#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs results
LOG="logs/server-$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$LOG" server.log
python - <<'PY'
import socket
s=socket.socket();s.settimeout(2)
try:
 if s.connect_ex(('127.0.0.1',30123))==0:raise SystemExit('Port 30123 already has a listener')
finally:s.close()
PY
nohup bash bootstrap.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > results/server.pid
for ((T=0; T<${LOAD_TIMEOUT:-900}; T+=5)); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Server exited; see $LOG" >&2
    wait "$PID"
    exit 1
  fi
  if curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:30123/health > /dev/null 2>&1; then
    echo "Ready: PID=$PID; log=$LOG"
    exit 0
  fi
  sleep 5
done
kill -TERM "$PID" 2>/dev/null || true
echo "Readiness timeout; see $LOG" >&2
exit 1
