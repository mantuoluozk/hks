"""One warmup-bearing embedding screen per isolated process; no corpus evaluation."""
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
O=R/'results'/time.strftime('optimization-%Y%m%dT%H%M%SZ',time.gmtime());O.mkdir(parents=True)
(O/'source').mkdir()
for p in ['scripts/native_runtime.py','scripts/benchmark_native.py','scripts/native_smoke.py','scripts/screen_optimization.py','run-scnet.sh']:
 (O/'source'/Path(p).name).write_bytes((R/p).read_bytes())
(R/'results/optimization-current.json').write_text(json.dumps({'path':str(O),'pid':os.getpid()}))
ref='/root/private_data/hks/models/Trident-Qwen3VL-2B/results/scnet/validation-20261026T114901Z/publisher_reference/summary.json'
def state(**kw): (O/'state.json').write_text(json.dumps(kw,indent=2))
for name,mode,mp in [('baseline','eager','1'),('inprocess','eager','0'),('piecewise-inprocess','piecewise','0')]:
 d=O/name;d.mkdir()
 env=dict(os.environ,RUN_RESULTS_DIR=str(d),REFERENCE_RESULTS=ref,RUN_BENCHMARK='1',TRIDENT_SCREEN='1',TRIDENT_MODE=mode,VLLM_ENABLE_V1_MULTIPROCESSING=mp)
 (d/'settings.json').write_text(json.dumps({k:env[k] for k in ['REFERENCE_RESULTS','TRIDENT_MODE','VLLM_ENABLE_V1_MULTIPROCESSING','TRIDENT_SCREEN']},indent=2))
 state(status='running',candidate=name)
 try:
  with (d/'run.log').open('w') as f:
   p=subprocess.Popen(['bash','run-scnet.sh','python','-u','scripts/native_smoke.py'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   (d/'process.pid').write_text(str(p.pid))
   try: code=p.wait(timeout=1200)
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=60);raise
  (d/'exit.json').write_text(json.dumps({'exit_code':code}))
  if code and name=='baseline':raise RuntimeError('baseline failed')
 except Exception as e:
  state(status='failed',candidate=name,error=repr(e));raise
 time.sleep(5)
state(status='completed')
