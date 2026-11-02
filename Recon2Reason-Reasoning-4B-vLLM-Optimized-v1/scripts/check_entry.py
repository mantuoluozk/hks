"""Exercise the released launch, environment, smoke and owned stop entries."""
import json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'results'/time.strftime('entry-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir()
def listening():
 with socket.socket() as s:return s.connect_ex(('127.0.0.1',30123))==0
assert not listening(),'Port occupied; no action'
pidfile=ROOT/'results/server.pid'
if pidfile.exists():pidfile.unlink()
report={'commands':[],'success':False}
try:
 for cmd in ['run.sh','env_check.sh','smoke.sh']:
  with (OUT/(cmd+'.log')).open('w') as f:
   result=subprocess.run(['bash',cmd],stdout=f,stderr=subprocess.STDOUT,timeout=1000)
  report['commands'].append({'command':cmd,'returncode':result.returncode})
  result.check_returncode()
 report['success']=True
finally:
 if pidfile.exists():
  with (OUT/'stop.sh.log').open('w') as f:subprocess.run(['bash','stop.sh'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
 for _ in range(120):
  if not listening():break
  time.sleep(1)
 report['port_closed']=not listening()
 report['success']=report['success'] and report['port_closed']
 (OUT/'summary.json').write_text(json.dumps(report,indent=2))
 (ROOT/'results/entry-current.json').write_text(json.dumps({'path':str(OUT)}))
 print(json.dumps(report,indent=2),flush=True)
assert report['success']
