"""Select a passing screen by total elapsed time, then run one full embedding process."""
import json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];os.chdir(R)
screen=Path(json.loads((R/'results/optimization-current.json').read_text())['path'])
while True:
 p=screen/'combined-state.json'
 if p.exists() and json.loads(p.read_text())['status']=='completed':break
 time.sleep(5)
baseline_dir=screen/'baseline'
if not baseline_dir.exists():baseline_dir=R/'results/optimization-20261102T080346Z/baseline'
baseline=json.loads((baseline_dir/'performance/summary.json').read_text())['cells']
candidates=[];rejected=[]
for name in ['inprocess','piecewise-inprocess','inprocess-cpu1','inprocess-cpu2','piecewise-best-cpu']:
 d=screen/name
 try:
  assert json.loads((d/'exit.json').read_text())['exit_code']==0
  functional=json.loads((d/'summary.json').read_text());assert functional['ranking']==json.loads((baseline_dir/'summary.json').read_text())['ranking']
  assert json.loads((d/'api-summary.json').read_text())['passed']
  cells=json.loads((d/'performance/summary.json').read_text())['cells'];assert len(cells)==12
  for a,b in zip(baseline,cells):
   assert (a['fixture'],a['batch_size'],a['input_tokens'])==(b['fixture'],b['batch_size'],b['input_tokens'])
   assert a['sample_prompt_token_ids']==b['sample_prompt_token_ids']
   assert np.min(np.sum(np.array(a['sample_vectors'])*np.array(b['sample_vectors']),axis=1))>0.99
  elapsed=sum(c['elapsed_seconds'] for c in cells)
  candidates.append({'name':name,'elapsed_seconds':elapsed,'aggregate_rps':768/elapsed})
 except Exception as exc: rejected.append({'name':name,'reason':repr(exc)})
assert candidates,'No eligible candidate'
selected=min(candidates,key=lambda c:c['elapsed_seconds'])['name']
settings=json.loads((screen/selected/'settings.json').read_text())
settings.setdefault('TRIDENT_CPU_THREADS','4')
O=R/'results'/time.strftime('final-%Y%m%dT%H%M%SZ',time.gmtime());O.mkdir();(O/'source').mkdir()
selection={'criterion':'Minimum sum of elapsed LLM.embed calls across all 12 screen cells; same 64 requests per cell. Single-screen selection, no confidence intervals.','selected':selected,'settings':settings,'candidates':candidates,'rejected':rejected,'screen':screen.name}
(O/'selection.json').write_text(json.dumps(selection,indent=2))
for rel in ['scripts/native_runtime.py','scripts/native_smoke.py','scripts/benchmark_native.py','scripts/final_validation.py','run-scnet.sh']:
 (O/'source'/Path(rel).name).write_bytes((R/rel).read_bytes())
(R/'results/final-current.json').write_text(json.dumps({'path':str(O),'controller_pid':os.getpid()}))
env=dict(os.environ,**{k:v for k,v in settings.items() if k in ['TRIDENT_MODE','VLLM_ENABLE_V1_MULTIPROCESSING','TRIDENT_CPU_THREADS']},TRIDENT_SCREEN='0',RUN_BENCHMARK='1',RUN_RESULTS_DIR=str(O),REFERENCE_RESULTS='/root/private_data/hks/models/Trident-Qwen3VL-2B/results/scnet/validation-20261026T114901Z/publisher_reference/summary.json')
(O/'state.json').write_text(json.dumps({'status':'running','selected':selected}))
with (O/'run.log').open('w') as f:
 proc=subprocess.Popen(['bash','run-scnet.sh','python','-u','scripts/native_smoke.py'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (O/'process.pid').write_text(str(proc.pid))
 try:code=proc.wait(timeout=1200)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);code=124
(O/'state.json').write_text(json.dumps({'status':'completed' if code==0 else 'failed','exit_code':code,'selected':selected,'independent_process_reproduction':None}))
raise SystemExit(code)
