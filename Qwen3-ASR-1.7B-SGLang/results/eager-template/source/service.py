"""Own a fresh service process group and health-check before reporting readiness."""
import json,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=Path(os.environ['RUN_RESULTS_DIR'])
port=int(os.environ['PORT'])
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0,'Port occupied'
command=[sys.executable,'-m','sglang.launch_server']+sys.argv[1:]
(out/'command.json').write_text(json.dumps(command,indent=2))
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
proc=None

def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
 with (out/'server.log').open('w') as log:
  proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (out/'server.pid').write_text(str(proc.pid))
  (root/'results/server-current.json').write_text(json.dumps({'path':str(out),'pid':proc.pid,'owner_pid':os.getpid()}))
  for _ in range(360):
   if proc.poll() is not None:raise RuntimeError(f'Service exited {proc.returncode}')
   try:
    with op.open(f'http://127.0.0.1:{port}/health',timeout=2) as res:assert res.status==200
    with op.open(f'http://127.0.0.1:{port}/v1/models',timeout=5) as res:models=json.load(res)
    assert any(x['id']==os.environ['SERVED_NAME'] for x in models['data'])
    (out/'models.json').write_text(json.dumps(models));(out/'ready.json').write_text(json.dumps({'passed':True}));print('SERVICE_READY',flush=True);break
   except Exception:time.sleep(5)
  else:raise TimeoutError('Readiness exceeded 30 minutes')
  code=proc.wait()
  if code not in [0,-15]:raise RuntimeError(f'Service exited {code}')
finally:
 if proc is not None:
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:proc.wait(timeout=60)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
