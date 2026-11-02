#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python - <<'PY'
import os,signal,time
from pathlib import Path
p=Path('results/service.pid')
if not p.exists():
    print('No owned service PID file');raise SystemExit(0)
pid=int(p.read_text())
proc=Path(f'/proc/{pid}')
if not proc.exists():
    p.unlink();print('Owned service already stopped');raise SystemExit(0)
cmd=(proc/'cmdline').read_bytes()
assert b'vllm.entrypoints.openai.api_server' in cmd and b'kimi-vl-a3b-instruct' in cmd, 'PID does not identify this model service'
os.kill(pid,signal.SIGTERM)
for _ in range(120):
    try:
        stopped=') Z ' in (proc/'stat').read_text()
    except FileNotFoundError:
        stopped=True
    if stopped:
        p.unlink();print('Owned service stopped');break
    time.sleep(.5)
else:
    raise RuntimeError('Service has not exited after 60 seconds; inspect its log')
PY
