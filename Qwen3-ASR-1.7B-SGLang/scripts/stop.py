"""Validate ownership and stop only this model's current service group."""
import json,os,signal,socket,time,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
info=json.loads((root/'results/server-current.json').read_text())
pid=info['pid'];p=Path('/proc')/str(pid)
assert (p/'cwd').resolve()==root
cmd=(p/'cmdline').read_bytes()
assert b'sglang.launch_server' in cmd and b'qwen3-asr-sglang' in cmd
assert os.getpgid(pid)==pid
intent={'pid':pid,'pgid':pid,'signal':'SIGTERM','phase':'requested','command_sha256':hashlib.sha256(cmd).hexdigest(),'requested_unix':time.time()}
target=Path(info['path'])/'stop-request.json';target.write_text(json.dumps(intent,indent=2)+'\n')
# Ask the API parent to drain; signalling every child simultaneously causes the
# native worker manager to interpret expected child termination as a crash.
os.kill(pid,signal.SIGTERM)
for _ in range(90):
    with socket.socket() as s:busy=s.connect_ex(('127.0.0.1',30129))==0
    try:gone=not p.exists() or (p/'stat').read_text().split()[2]=='Z'
    except FileNotFoundError:gone=True
    if not busy and gone:
        intent.update(phase='stopped',port_free=True,finished_unix=time.time());target.write_text(json.dumps(intent,indent=2)+'\n')
        print('OWNED_SERVICE_STOPPED');break
    time.sleep(1)
else:raise TimeoutError('Owned service did not stop')
