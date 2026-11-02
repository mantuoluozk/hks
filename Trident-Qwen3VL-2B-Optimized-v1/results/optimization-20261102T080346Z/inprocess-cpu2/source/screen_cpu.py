"""Screen one- and two-thread preprocessing, once each, after the execution screen."""
import json,os,subprocess,time,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
O=Path(json.loads((R/'results/optimization-current.json').read_text())['path'])
while True:
 state=json.loads((O/'state.json').read_text())
 if state['status']=='completed':break
 if state['status']=='failed':raise RuntimeError(state)
 time.sleep(5)
for threads in [1,2]:
 d=O/f'inprocess-cpu{threads}';d.mkdir();(d/'source').mkdir()
 for p in ['scripts/native_runtime.py','scripts/benchmark_native.py','scripts/native_smoke.py','scripts/screen_cpu.py','run-scnet.sh']:
  (d/'source'/Path(p).name).write_bytes((R/p).read_bytes())
 env=dict(os.environ,TRIDENT_CPU_THREADS=str(threads),VLLM_ENABLE_V1_MULTIPROCESSING='0',TRIDENT_MODE='eager',TRIDENT_SCREEN='1',RUN_BENCHMARK='1',RUN_RESULTS_DIR=str(d),REFERENCE_RESULTS='/root/private_data/hks/models/Trident-Qwen3VL-2B/results/scnet/validation-20261026T114901Z/publisher_reference/summary.json')
 (d/'settings.json').write_text(json.dumps({k:env[k] for k in ['TRIDENT_CPU_THREADS','VLLM_ENABLE_V1_MULTIPROCESSING','TRIDENT_MODE']},indent=2))
 (O/'cpu-state.json').write_text(json.dumps({'status':'running','candidate':d.name,'controller_pid':os.getpid()}))
 with (d/'run.log').open('w') as f:
  p=subprocess.Popen(['bash','run-scnet.sh','python','-u','scripts/native_smoke.py'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.pid').write_text(str(p.pid))
  try:code=p.wait(timeout=900)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=60);code=124
 (d/'exit.json').write_text(json.dumps({'exit_code':code}));time.sleep(5)
(O/'cpu-state.json').write_text(json.dumps({'status':'completed'}))
