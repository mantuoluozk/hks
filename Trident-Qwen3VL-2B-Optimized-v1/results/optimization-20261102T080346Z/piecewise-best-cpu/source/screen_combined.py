"""Test the best CPU thread candidate together with compiled execution, once."""
import json,os,subprocess,time,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
O=Path(json.loads((R/'results/optimization-current.json').read_text())['path'])
while not (O/'cpu-state.json').exists() or json.loads((O/'cpu-state.json').read_text())['status']!='completed':time.sleep(5)
status=O/'combined-state.json'
def elapsed(name):
 p=O/name
 if not (p/'exit.json').exists() or json.loads((p/'exit.json').read_text())['exit_code']!=0:return float('inf')
 return sum(c['elapsed_seconds'] for c in json.loads((p/'performance/summary.json').read_text())['cells'])
best=min([1,2,4],key=lambda t:elapsed('inprocess' if t==4 else f'inprocess-cpu{t}'))
if best==4 or elapsed('piecewise-inprocess')==float('inf'):
 status.write_text(json.dumps({'status':'completed','skipped':'No better CPU candidate or compiled candidate failed'}));raise SystemExit(0)
d=O/'piecewise-best-cpu';d.mkdir();(d/'source').mkdir()
for rel in ['scripts/native_runtime.py','scripts/native_smoke.py','scripts/benchmark_native.py','scripts/screen_combined.py','run-scnet.sh']:(d/'source'/Path(rel).name).write_bytes((R/rel).read_bytes())
settings={'TRIDENT_CPU_THREADS':str(best),'VLLM_ENABLE_V1_MULTIPROCESSING':'0','TRIDENT_MODE':'piecewise'}
(d/'settings.json').write_text(json.dumps(settings,indent=2));status.write_text(json.dumps({'status':'running','settings':settings,'pid':os.getpid()}))
env=dict(os.environ,**settings,TRIDENT_SCREEN='1',RUN_BENCHMARK='1',RUN_RESULTS_DIR=str(d),REFERENCE_RESULTS='/root/private_data/hks/models/Trident-Qwen3VL-2B/results/scnet/validation-20261026T114901Z/publisher_reference/summary.json')
with (d/'run.log').open('w') as f:
 p=subprocess.Popen(['bash','run-scnet.sh','python','-u','scripts/native_smoke.py'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (d/'process.pid').write_text(str(p.pid))
 try:code=p.wait(timeout=900)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=60);code=124
(d/'exit.json').write_text(json.dumps({'exit_code':code}));status.write_text(json.dumps({'status':'completed','exit_code':code}))
