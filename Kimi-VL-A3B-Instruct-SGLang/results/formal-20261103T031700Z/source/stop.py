"""Stop only the PID recorded by this model's standard entrypoint."""
import os
from pathlib import Path
import signal
import time
root=Path(__file__).resolve().parents[1]
pid=int((root/'results/runtime/server.pid').read_text())
proc=Path('/proc')/str(pid)
assert proc.exists(), 'Recorded service is not running'
assert (proc/'cwd').resolve()==root
assert b'scripts/launch.py' in (proc/'cmdline').read_bytes()
if os.getpgid(pid)==pid:
    os.killpg(pid,signal.SIGTERM)
else:
    os.kill(pid,signal.SIGTERM)
for _ in range(60):
    if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':
        print('Owned model service stopped');break
    time.sleep(1)
else:
    raise TimeoutError('Owned service did not stop within 60 seconds')
