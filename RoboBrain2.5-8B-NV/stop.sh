#!/usr/bin/env bash
# Stops only the recorded PID when its model, name and port match this project.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source env.sh
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-results}"
python - <<'PY'
import os,signal,time
from pathlib import Path
record=Path(os.environ['RUN_RESULTS_DIR'],'server.pid')
if not record.exists():raise SystemExit('No recorded server PID; nothing stopped.')
pid=int(record.read_text());proc=Path(f'/proc/{pid}/cmdline')
if not proc.exists():raise SystemExit('Recorded server already stopped.')
cmd=proc.read_bytes().split(b'\0')
def has_pair(flag,value):
    f=flag.encode();v=value.encode()
    return any(cmd[i]==f and cmd[i+1]==v for i in range(len(cmd)-1))
assert b'sglang.launch_server' in cmd,'Unexpected process; refusing to stop'
for key,flag in [('MODEL_PATH','--model-path'),('SERVED_NAME','--served-model-name'),('PORT','--port')]:
    assert has_pair(flag,os.environ[key]),f'PID does not match {key}; refusing to stop'
os.kill(pid,signal.SIGTERM)
for _ in range(60):
    if not proc.exists():break
    time.sleep(1)
else:raise SystemExit('Server has not exited; inspect before restarting.')
print('Stopped this model server:',pid)
PY
