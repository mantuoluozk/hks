import os,subprocess,time,json,urllib.request
from pathlib import Path
out=Path(os.environ['PROBE_OUT'])
env=dict(os.environ)
log=(out/'server.log').open('w')
p=subprocess.Popen(['bash','serve-formal.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'server.pid').write_text(str(p.pid))
try:
 for _ in range(180):
  if p.poll() is not None: raise RuntimeError('server exited')
  try:
   urllib.request.urlopen('http://127.0.0.1:30004/health',timeout=2);break
  except Exception: time.sleep(5)
 else: raise RuntimeError('load timeout')
 args=['python','scripts/benchmark_valid_vocab.py','--backend','vllm-chat','--host','127.0.0.1','--port','30004','--model',str(Path.cwd()/'weights'),'--served-model-name','kimi-vl-a3b-instruct','--tokenizer',str(Path.cwd()/'weights'),'--dataset-name','image','--image-count','1','--image-resolution','1080p','--random-input-len','4096','--random-output-len','1024','--random-range-ratio','1','--max-concurrency','1','--num-prompts','2','--seed','1','--extra-request-body',json.dumps({'stream_options':{'include_usage':True}}),'--output-details','--output-file',str(out/'probe.jsonl')]
 with (out/'probe.log').open('w') as f: subprocess.run(['bash','run-scnet.sh']+args,stdout=f,stderr=subprocess.STDOUT,timeout=600,check=True)
 (out/'state.json').write_text(json.dumps({'status':'completed','formal_result':False}))
except Exception as e:
 (out/'state.json').write_text(json.dumps({'status':'failed','error':repr(e)}))
finally:
 p.terminate()
 try:p.wait(timeout=90)
 except subprocess.TimeoutExpired:p.kill()
