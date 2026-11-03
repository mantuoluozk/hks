"""Two bounded candidates, one warmed short matrix each; preserve failed runs."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT/'results'/time.strftime('screen-%Y%m%dT%H%M%SZ',time.gmtime())
OUT.mkdir()
(ROOT/'results/current.json').write_text(json.dumps({'path':str(OUT),'pid':os.getpid()}))
(OUT/'source').mkdir()
for p in list((ROOT/'scripts').glob('*.py')) + list(ROOT.glob('*.sh')):
    shutil.copyfile(p,OUT/'source'/p.name)
def state(**kw):
    (OUT/'state.json').write_text(json.dumps(kw,indent=2))
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
proc=None
try:
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',30004)) != 0, 'Port occupied'
    for name, flags in [('eager',['--disable-cuda-graph']),('decode-graph',['--cuda-graph-bs-decode','1','2','4','8'])]:
        target=OUT/name;target.mkdir()
        command=['bash','serve-scnet.sh']+flags
        (target/'command.json').write_text(json.dumps(command))
        with (target/'server.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (target/'server.pid').write_text(str(proc.pid))
            state(stage='loading',candidate=name,pid=proc.pid)
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f'{name} server exit {proc.returncode}')
                try:
                    with op.open('http://127.0.0.1:30004/health',timeout=2) as r:
                        if r.status==200: break
                except Exception:
                    time.sleep(5)
            else: raise TimeoutError('Health timeout')
            for stage,script,extra in [('functional','smoke_http.py',[]),('short','bench_http.py',['--smoke-summary',str(target/'functional/summary.json')])]:
                state(stage=stage,candidate=name,pid=proc.pid)
                with (target/(stage+'.log')).open('w') as f:
                    subprocess.run(['bash','run-scnet.sh','python','-u','scripts/'+script,'--output',str(target/stage)]+extra,
                                   stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=60);proc=None
            time.sleep(3)
    state(stage='complete',formal_started=False)
except BaseException as error:
    state(stage='failed',error=repr(error),formal_started=False)
    raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
