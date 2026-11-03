"""Official-template image benchmark, one service, two repetitions per cell."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser=argparse.ArgumentParser()
parser.add_argument('--processes',type=int,choices=(1,2),default=1)
parser.add_argument('--launcher',default='run.sh')
args=parser.parse_args()
launcher=(ROOT/args.launcher).resolve()
assert launcher.is_relative_to(ROOT) and launcher.is_file()
OUT=ROOT/'results'/time.strftime('formal-%Y%m%dT%H%M%SZ',time.gmtime())
OUT.mkdir()
OUT.joinpath('source').mkdir()
for rel in (args.launcher,'scripts/run_formal_matrix.py','scripts/prepare_bench_alias.py','scripts/benchmark_valid_vocab.py','scripts/check_bench_row.py','verify_formal.py'):
 p=OUT/'source'/Path(rel).name;p.write_bytes((ROOT/rel).read_bytes())
(ROOT/'results/formal-current.json').write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
env=dict(os.environ,MAX_LEN='24576',MAX_RUNNING_REQUESTS='8',MEM_FRAC='0.80',
         MODEL_PATH=str(ROOT/'weights'),
         SERVED_NAME='kimi-vl-a3b-instruct',PORT='30004',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for k,v in {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items():env[k]=v
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def state(**kw):
 (OUT/'state.json').write_text(json.dumps({'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2))
def ready():
 try:
  with op.open('http://127.0.0.1:30004/health',timeout=3) as r:return r.status==200
 except Exception:return False
def command(args,log,timeout=7200):
 with log.open('w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
proc=None
probe=socket.socket()
try:busy=probe.connect_ex(('127.0.0.1',30004))==0
finally:probe.close()
if busy:
 state(status='failed_before_start',error='Port 30004 has a listener')
 raise RuntimeError('Port 30004 has a listener')
try:
 for process in range(1,args.processes+1):
  rd=OUT/f'process{process}';rd.mkdir()
  log=(rd/'server.log').open('w')
  proc=subprocess.Popen(['bash',str(launcher)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (rd/'server.pid').write_text(str(proc.pid))
  state(status='loading',process=process,server_pid=proc.pid)
  for _ in range(180):
   if proc.poll() is not None:raise RuntimeError('Server exited; see '+str(rd/'server.log'))
   if ready():break
   time.sleep(5)
  else:raise TimeoutError('Readiness timeout')
  command(['python','scripts/smoke_http.py','--output',str(rd/'functional')],rd/'smoke.log',900)
  state(status='ocrbench',process=process,server_pid=proc.pid,samples=1000)
  command(['python','scripts/eval_ocrbench.py','--smoke-summary',str(rd/'functional/summary.json'),
   '--output',str(rd/'ocrbench'),'--workers','2'],rd/'ocrbench.log',7200)
  command(['python','scripts/audit_ocrbench.py','--run',str(rd/'ocrbench')],rd/'ocrbench-audit.log',180)
  command(['python','scripts/prepare_bench_alias.py'],rd/'processor-alias.log',60)
  env['BENCH_INPUTS_DIR']=str(rd/'context-probe-inputs')
  command(['python','scripts/benchmark_valid_vocab.py','--backend','sglang-oai-chat',
   '--host','127.0.0.1','--port','30004','--model',env['MODEL_PATH'],
   '--served-model-name',env['SERVED_NAME'],'--tokenizer',env['MODEL_PATH'],
   '--dataset-name','image','--image-count','1','--image-resolution','1080p',
   '--random-input-len','16384','--random-output-len','32','--random-range-ratio','1',
   '--max-concurrency','1','--num-prompts','1','--seed','1',
   '--extra-request-body',json.dumps({'stream_options':{'include_usage':True}}),
   '--output-details','--output-file',str(rd/'context-probe.jsonl')],rd/'context-probe.log')
  probe_row=json.loads((rd/'context-probe.jsonl').read_text().splitlines()[-1])
  assert probe_row['completed']==1 and not any(probe_row['errors'])
  assert all(n+1024<=24576 for n in probe_row['input_lens']), probe_row['input_lens']
  for length in (4096,16384):
   for concurrency in (1,8,64):
    count=max(8,2*concurrency)
    for repeat in (1,2):
     tag=f'round{repeat}_{length}_1024_c{concurrency}'
     state(status='benchmarking',process=process,input_tokens=length,output_tokens=1024,concurrency=concurrency,repeat=repeat,requests=count)
     env['BENCH_INPUTS_DIR']=str(rd/(tag+'-inputs'))
     command(['python','scripts/benchmark_valid_vocab.py','--backend','sglang-oai-chat',
      '--host','127.0.0.1','--port','30004','--model',env['MODEL_PATH'],
      '--served-model-name',env['SERVED_NAME'],'--tokenizer',env['MODEL_PATH'],
      '--dataset-name','image','--image-count','1','--image-resolution','1080p',
      '--random-input-len',str(length),'--random-output-len','1024','--random-range-ratio','1',
      '--max-concurrency',str(concurrency),'--num-prompts',str(count),'--seed','1',
      '--extra-request-body',json.dumps({'stream_options':{'include_usage':True}}),
      '--output-details','--output-file',str(rd/f'round{repeat}.jsonl')],rd/f'{tag}.log')
     command(['python','scripts/check_bench_row.py',str(rd/f'round{repeat}.jsonl'),str(length),'1024',str(concurrency),str(count)],rd/f'{tag}.check.log',60)
  command(['bash','stop.sh'],rd/'stop.log',90)
  proc.wait(timeout=60);proc=None
  log.close();time.sleep(10)
 command(['python','verify_formal.py','--job',OUT.name,'--processes',str(args.processes)],OUT/'verification.log',120)
 state(status='completed',official_acceptance=None)
except Exception as exc:
 state(status='failed',error=repr(exc),official_acceptance=None)
 raise
finally:
 if proc and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=60)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
