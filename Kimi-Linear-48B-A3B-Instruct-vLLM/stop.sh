#!/usr/bin/env bash
# Stop only the owned vLLM process; never search for SGLang services.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
python3 - <<'PY'
import os,signal
from pathlib import Path
p=Path('results/server.pid')
if p.exists():
    pid=int(p.read_text());cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists():
        argv=cmd.read_bytes().split(b'\0')
        if b'vllm.entrypoints.openai.api_server' in argv and os.environ['MODEL_NAME'].encode() in argv:
            os.kill(pid,signal.SIGTERM);print('Stopped owned vLLM server',pid)
        else:print('PID is not the owned vLLM server; no action')
    else:print('Server already stopped')
else:print('No owned server PID file')
PY
