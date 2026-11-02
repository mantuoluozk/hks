import concurrent.futures as cf
import json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'results'/time.strftime('optimization-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir();(OUT/'source').mkdir()
for rel in ['bootstrap.sh','scripts/screen_graph.py','scripts/smoke_test.py','scripts/validate.py']:(OUT/'source'/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
(ROOT/'results/optimization-current.json').write_text(json.dumps({'path':str(OUT),'pid':os.getpid()}))
env=dict(os.environ,MAX_LEN='32768',MEM_FRAC='0.80',MAX_RUNNING_REQUESTS='64',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',EXECUTION_PLATFORM='SCNet Notebook')
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));proc=None

def state(**kw):(OUT/'state.json').write_text(json.dumps(kw,indent=2))
def ready():
 try:
  with op.open('http://127.0.0.1:30123/health',timeout=2) as r:return r.status==200
 except Exception:return False

def call(n):
 body={'model':'recon2reason-4b','messages':[{'role':'user','content':'Explain the properties of a red rectangle. '*n}],'temperature':0,'max_tokens':128,'ignore_eos':True}
 t=time.monotonic()
 with op.open(urllib.request.Request('http://127.0.0.1:30123/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=300) as r:data=json.load(r)
 dt=time.monotonic()-t
 assert data['usage']['completion_tokens']==128,data
 assert data['choices'][0]['message'].get('content'),data
 return {'seconds':dt,'request':body,'response':data}

assert not ready(),'port occupied'
try:
 summaries={}
 for name in ['eager','decode-graph']:
  rd=OUT/name;rd.mkdir()
  launcher=(ROOT/'bootstrap.sh').read_text().replace('cd -- "$(dirname -- "${BASH_SOURCE[0]}")"','')
  if name=='decode-graph':launcher=launcher.replace('--enforce-eager',"--compilation-config '"+json.dumps({'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8,16,32,64]},separators=(',',':'))+"'")
  (rd/'launcher.sh').write_text(launcher)
  with (rd/'server.log').open('w') as log:
   proc=subprocess.Popen(['bash','-c',launcher],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (rd/'server.pid').write_text(str(proc.pid));state(status='loading',candidate=name,pid=proc.pid)
   for _ in range(240):
    if proc.poll() is not None:raise RuntimeError(name+' exited')
    if ready():break
    time.sleep(5)
   else:raise TimeoutError('load')
   with (rd/'smoke.log').open('w') as f:subprocess.run(['python','scripts/smoke_test.py'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
   cells=[]
   with (rd/'requests.jsonl').open('w') as f:
    for length in [32,256]:
     row=call(length);f.write(json.dumps({'kind':'warmup',**row})+'\n');f.flush()
     for concurrency in [1,2,4]:
      state(status='benchmarking',candidate=name,prompt_repeats=length,concurrency=concurrency)
      t=time.monotonic()
      with cf.ThreadPoolExecutor(concurrency) as pool:rows=list(pool.map(call,[length]*8))
      elapsed=time.monotonic()-t
      for row in rows:f.write(json.dumps({'kind':'measured','prompt_repeats':length,'concurrency':concurrency,**row})+'\n')
      f.flush();cells.append({'prompt_repeats':length,'concurrency':concurrency,'requests':8,'elapsed':elapsed,'output_tokens_per_second':1024/elapsed})
   (rd/'summary.json').write_text(json.dumps(cells,indent=2));summaries[name]=cells
   proc.terminate();proc.wait(timeout=120);proc=None
  time.sleep(5)
 comp=[]
 for a,b in zip(summaries['eager'],summaries['decode-graph']):comp.append({'prompt_repeats':a['prompt_repeats'],'concurrency':a['concurrency'],'baseline_tps':a['output_tokens_per_second'],'candidate_tps':b['output_tokens_per_second'],'gain_pct':(b['output_tokens_per_second']/a['output_tokens_per_second']-1)*100})
 aa=[json.loads(x) for x in (OUT/'eager/requests.jsonl').read_text().splitlines()];bb=[json.loads(x) for x in (OUT/'decode-graph/requests.jsonl').read_text().splitlines()]
 same=all(a['request']==b['request'] and a['response']['usage']==b['response']['usage'] and a['response']['choices'][0]['message']['content']==b['response']['choices'][0]['message']['content'] for a,b in zip(aa,bb)) and len(aa)==len(bb)==50
 (OUT/'comparison.json').write_text(json.dumps({'cells':comp,'requests_and_text_equal':same},indent=2));state(status='completed',requests_and_text_equal=same)
except Exception as e:state(status='failed',error=repr(e));raise
finally:
 if proc and proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=120)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM)
