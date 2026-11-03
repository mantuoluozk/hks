"""Two execution candidates, one warmed short pass each; no automatic formal run."""
import json,os,signal,socket,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
gate=json.loads((ROOT/'results/client-compat-v2/check.json').read_text());assert gate['passed']
OUT=ROOT/'results'/time.strftime('screen-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir()
(ROOT/'results/screen-current.json').write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
(OUT/'source').mkdir()
for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.sh')):shutil.copyfile(p,OUT/'source'/p.name)
def state(stage,**extra):(OUT/'state.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
proc=None
try:
    for mode in ['eager','graph']:
        target=OUT/mode;target.mkdir()
        env=dict(os.environ,EXECUTION_MODE=mode,RUN_RESULTS_DIR=str(target),SMOKE_OUTPUT_DIR=str(target/'functional'))
        with (target/'launcher.log').open('w') as log:
            proc=subprocess.Popen(['bash','run.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state('loading',candidate=mode,owner_pid=proc.pid)
            for _ in range(360):
                if proc.poll() is not None:raise RuntimeError(f'{mode} launcher exit {proc.returncode}')
                if (target/'ready.json').exists():break
                time.sleep(5)
            else:raise TimeoutError('Readiness timeout')
            for phase,cmd in [('functional',['bash','smoke.sh']),('short',['python','scripts/short.py','--output',str(target/'short')])]:
                state(phase,candidate=mode,owner_pid=proc.pid)
                with (target/(phase+'-driver.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
            with (target/'stop.log').open('w') as f:subprocess.run(['bash','stop.sh'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=100)
            assert proc.wait(timeout=90)==0
            proc=None
    state('completed',formal_started=False)
except BaseException as e:
    state('failed',error=repr(e),formal_started=False);raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=100)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
