"""Initial native-route readiness, eight functional cases, and one short warmup screen."""
import json,os,subprocess,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R);O=R/'results/initial'
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def state(**kw):(O/'state.json').write_text(json.dumps(kw,indent=2))
try:
 state(status='loading')
 for _ in range(360):
  try:
   with op.open('http://127.0.0.1:30126/health',timeout=2) as r:
    if r.status==200:break
  except Exception:time.sleep(5)
 else:raise TimeoutError('Native service readiness')
 state(status='functional')
 with (O/'smoke.log').open('w') as f:
  subprocess.run(['python','scripts/smoke_test.py'],env=dict(os.environ,SMOKE_OUTPUT_DIR=str(O/'functional')),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
 state(status='short_screen')
 with (O/'screen.log').open('w') as f:subprocess.run(['python','scripts/short_screen.py'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
 state(status='completed')
except Exception as exc:state(status='failed',error=repr(exc));raise
