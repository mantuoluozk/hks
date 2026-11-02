#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
python - <<'PY'
from pathlib import Path
import os,signal,socket,time
p=Path('results/server.pid')
if not p.exists():raise SystemExit('No standard-entry service PID; task controllers manage their own PID files')
pid=int(p.read_text());path=Path(f'/proc/{pid}/cmdline')
if not path.exists():print('Service already stopped');raise SystemExit(0)
parts=path.read_bytes().split(b'\0')
assert b'vllm.entrypoints.openai.api_server' in parts and os.environ['MODEL_PATH'].encode() in parts and os.environ['SERVED_NAME'].encode() in parts,str(pid)
assert b'--port' in parts and parts[parts.index(b'--port')+1]==os.environ['PORT'].encode(),str(pid)
os.kill(pid,signal.SIGTERM)
for _ in range(90):
 with socket.socket() as s:
  if s.connect_ex(('127.0.0.1',int(os.environ['PORT'])))!=0:print('Stopped',pid);break
 time.sleep(1)
else:raise SystemExit('Stop timed out')
PY
