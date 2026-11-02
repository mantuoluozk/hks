"""Two independent HTTP service processes; synthetic functionality and performance."""
import concurrent.futures as cf
import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT=ROOT/'results'/time.strftime('validation-%Y%m%dT%H%M%SZ',time.gmtime())
OUT.mkdir()
OP=urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE='http://127.0.0.1:30123'
def ready():
 try:
  with OP.open(BASE+'/health',timeout=3) as r: return r.status==200
 except Exception: return False
def call(n):
 body={'model':'recon2reason-4b','messages':[{'role':'user','content':('Explain the properties of a red rectangle. '*n)}],'temperature':0,'max_tokens':128,'ignore_eos':True}
 t=time.monotonic()
 with OP.open(urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=300) as r: data=json.load(r)
 dt=time.monotonic()-t
 assert data['usage']['completion_tokens']==128,data
 assert data['choices'][0]['message'].get('content'),data
 return {'seconds':dt,'request':body,'response':data}
owned=None
rounds=[]
if ready():raise RuntimeError('Port 30123 is already serving; stop that service before running bench.sh')
try:
 for roundno in (1,2):
  rd=OUT/f'round{roundno}';rd.mkdir()
  if roundno in (1,2):
   log=(rd/'server.log').open('w')
   proc=subprocess.Popen(['bash','bootstrap.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   owned=proc.pid
  (rd/'server.pid').write_text(str(owned))
  for _ in range(180):
   if proc.poll() is not None:raise RuntimeError(f'service exited; see {rd}/server.log')
   if ready():break
   os.kill(owned,0);time.sleep(5)
  else:raise RuntimeError('service readiness timeout')
  env=dict(os.environ,PORT='30123',SERVED_NAME='recon2reason-4b',EXECUTION_PLATFORM='SCNet Notebook')
  with (rd/'smoke.log').open('w') as f: subprocess.run(['python','scripts/smoke_test.py'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  cells=[]
  with (rd/'requests.jsonl').open('w') as evidence:
   for length in (32,256):
    warm=call(length);evidence.write(json.dumps({'kind':'warmup',**warm})+'\n');evidence.flush()
    for concurrency in (1,2,4):
     t=time.monotonic()
     with cf.ThreadPoolExecutor(concurrency) as pool: rows=list(pool.map(call,[length]*8))
     elapsed=time.monotonic()-t
     for row in rows:evidence.write(json.dumps({'kind':'measured','prompt_repeats':length,'concurrency':concurrency,**row})+'\n')
     evidence.flush()
     cells.append({'prompt_repeats':length,'concurrency':concurrency,'requests':8,'elapsed':elapsed,'mean_seconds':sum(x['seconds'] for x in rows)/8,'output_tokens_per_second':1024/elapsed})
  (rd/'summary.json').write_text(json.dumps(cells,indent=2));rounds.append(cells)
  os.kill(owned,signal.SIGTERM)
  proc.wait(timeout=120)
  for _ in range(60):
   if not ready():break
   time.sleep(2)
  time.sleep(10)
  owned=None
 deviations=[{'prompt_repeats':a['prompt_repeats'],'concurrency':a['concurrency'],**{k:abs(b[k]-a[k])/a[k] for k in ('mean_seconds','output_tokens_per_second')}} for a,b in zip(*rounds)]
 summary={'rounds':rounds,'deviations':deviations,'performance_reproduced':all(d[k]<=.2 for d in deviations for k in ('mean_seconds','output_tokens_per_second')),'official_acceptance':None,'accuracy':'exempt under supplied competition rule; synthetic smoke is not accuracy'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
finally:
 if owned:
  try:os.kill(owned,signal.SIGTERM)
  except ProcessLookupError:pass
 with (OUT/'backup.log').open('w') as f:subprocess.run(['python','/root/private_data/hks/platform/scnet-notebook/backup_shared.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
