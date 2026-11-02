#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p results logs
python - <<'PY'
import os,socket
with socket.socket() as s:
 if s.connect_ex(('127.0.0.1',int(os.environ['PORT'])))==0:raise SystemExit('Service port already occupied')
PY
log="logs/server-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash serve.sh > "$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > results/server.pid
for ((t=0;t<1800;t+=5)); do
 if ! kill -0 "$pid" 2>/dev/null; then echo "Server exited; see $log" >&2; exit 1; fi
 if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "Ready: PID=$pid log=$log"; exit 0; fi
 sleep 5
done
kill -TERM "$pid" 2>/dev/null || true
echo "Readiness timeout; see $log" >&2
exit 1
