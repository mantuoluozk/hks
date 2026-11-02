#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python - <<'PY'
import os,signal
from pathlib import Path
p=Path('results/server.pid')
if p.exists():
 pid=int(p.read_text())
 cmd=Path(f'/proc/{pid}/cmdline')
 if cmd.exists():
  text=cmd.read_bytes()
  if b'vllm.entrypoints.openai.api_server' in text and b'recon2reason-4b' in text:
   os.kill(pid,signal.SIGTERM)
   print('Stopped model server',pid)
  else:print('PID is no longer this model server; no action')
 else:print('Server already stopped')
else:print('No owned server PID file')
PY
