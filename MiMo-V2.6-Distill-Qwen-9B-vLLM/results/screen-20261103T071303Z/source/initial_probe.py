"""Gate initial functionality before one warmed eager short pass; always stop."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/initial'
def state(stage,**extra):(out/'state.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
try:
    state('waiting_for_health')
    for _ in range(360):
        if (out/'ready.json').exists():break
        info=json.loads((ROOT/'results/server-current.json').read_text())
        p=Path('/proc')/str(info['owner_pid'])
        if not p.exists() or (p/'stat').read_text().split()[2]=='Z':raise RuntimeError('Service launcher exited')
        time.sleep(5)
    else:raise TimeoutError('Readiness timeout')
    for name,command in [('functional',['bash','smoke.sh']),('short',['python','scripts/short.py','--output',str(out/'short')])]:
        state(name)
        with (out/(name+'-driver.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,SMOKE_OUTPUT_DIR=str(out/'functional')),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    state('completed')
except BaseException as e:
    state('failed',error=repr(e));raise
finally:
    with (out/'stop.log').open('w') as log:subprocess.run(['python','scripts/stop.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=100)
