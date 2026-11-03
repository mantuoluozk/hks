"""Validate ownership and stop only this model's current service group."""
import json,os,signal,socket,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
info=json.loads((root/'results/server-current.json').read_text())
pid=info['pid'];p=Path('/proc')/str(pid)
assert (p/'cwd').resolve()==root
cmd=(p/'cmdline').read_bytes()
assert b'sglang.launch_server' in cmd and b'qwen3-asr-sglang' in cmd
assert os.getpgid(pid)==pid
os.killpg(pid,signal.SIGTERM)
for _ in range(90):
    with socket.socket() as s:busy=s.connect_ex(('127.0.0.1',30129))==0
    if not busy and (not p.exists() or (p/'stat').read_text().split()[2]=='Z'):
        print('OWNED_SERVICE_STOPPED');break
    time.sleep(1)
else:raise TimeoutError('Owned service did not stop')
