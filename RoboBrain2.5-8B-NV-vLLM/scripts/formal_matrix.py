"""One RoboBrain native-vLLM service, 608 measured generation requests."""
import json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
O=R/'results'/time.strftime('formal-%Y%m%dT%H%M%SZ',time.gmtime());O.mkdir();(O/'source').mkdir()
for rel in ['serve.sh','env.sh','run-scnet.sh','scripts/formal_matrix.py','scripts/check_bench_row.py','scripts/smoke_test.py','scripts/prepare_bench_alias.py','scripts/check_weights.py','verify.py']:(O/'source'/Path(rel).name).write_bytes((R/rel).read_bytes())
(R/'results/formal-current.json').write_text(json.dumps({'path':str(O),'pid':os.getpid()}))
env=dict(os.environ,SMOKE_OUTPUT_DIR=str(O/'functional'),RUN_RESULTS_DIR=str(O/'weights'))
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def state(**kw):(O/'state.json').write_text(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2))
def run(argv,log,timeout=7200):
 with log.open('w') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)
def metrics(name):
 with op.open('http://127.0.0.1:30126/metrics',timeout=20) as r:(O/name).write_bytes(r.read())
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',30126))!=0,'Port busy; do not duplicate service'
proc=None
try:
 with (O/'server.log').open('w') as f:
  proc=subprocess.Popen(['bash','serve.sh'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (O/'server.pid').write_text(str(proc.pid));state(status='loading',server_pid=proc.pid)
  for _ in range(360):
   if proc.poll() is not None:raise RuntimeError('Service exited')
   try:
    with op.open('http://127.0.0.1:30126/health',timeout=2) as response:
     if response.status==200:break
   except Exception:time.sleep(5)
  else:raise TimeoutError('Readiness timeout')
  state(status='functional')
  run(['python','scripts/smoke_test.py'],O/'smoke.log',900)
  run(['python','scripts/prepare_bench_alias.py','scripts/check_weights.py','verify.py'],O/'alias.log',60)
  metrics('metrics-before.prom')
  for length in [4096,16384]:
   for concurrency in [1,8,64]:
    count=max(8,2*concurrency)
    for repeat in [1,2]:
     tag=f'round{repeat}_{length}_1024_c{concurrency}';state(status='benchmarking',input_length=length,concurrency=concurrency,repeat=repeat,requests=count)
     run(['python','-m','sglang.benchmark.serving','--backend','vllm-chat','--host','127.0.0.1','--port','30126','--model',env['MODEL_PATH'],'--served-model-name',env['SERVED_NAME'],'--tokenizer',env['MODEL_PATH'],'--dataset-name','image','--image-count','1','--image-resolution','1080p','--random-input-len',str(length),'--random-output-len','1024','--random-range-ratio','1','--max-concurrency',str(concurrency),'--num-prompts',str(count),'--seed','1','--extra-request-body',json.dumps({'stream_options':{'include_usage':True}}),'--output-details','--output-file',str(O/f'round{repeat}.jsonl')],O/(tag+'.log'))
     run(['python','scripts/check_bench_row.py',str(O/f'round{repeat}.jsonl'),str(length),'1024',str(concurrency),str(count)],O/(tag+'.check.log'),60)
  metrics('metrics-after.prom')
  proc.terminate();proc.wait(timeout=120);proc=None
 run(['python','verify.py','--selfcheck'],O/'verification.log',120)
 state(status='completed',official_acceptance=None,independent_process_reproduction=None)
except Exception as e:state(status='failed',error=repr(e));raise
finally:
 if proc and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=120)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
