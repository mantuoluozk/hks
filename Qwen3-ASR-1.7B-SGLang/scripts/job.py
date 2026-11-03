"""Own one final service; run requested audio stages serially, then stop it."""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['formal','bench','accuracy','official'],required=True);a=p.parse_args()
selection=json.loads((ROOT/'results/selection.json').read_text());assert selection['passed']
OUT=ROOT/'results'/time.strftime(a.stage+'-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir()
(ROOT/'results'/(a.stage+'-current.json')).write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
(OUT/'source').mkdir()
for f in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.sh')):shutil.copyfile(f,OUT/'source'/f.name)
env=dict(os.environ,EXECUTION_MODE=selection['mode'],RUN_RESULTS_DIR=str(OUT),SMOKE_OUTPUT_DIR=str(OUT/'functional'),ASR_USE_RUNNING_SERVICE='1')
def state(stage,**kw):(OUT/'state.json').write_text(json.dumps({'stage':stage,**kw},indent=2))
def command(cmd,name,timeout,check=True):
    with (OUT/(name+'.log')).open('w') as f:return subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=check)
proc=None
try:
    if a.stage in ['formal','accuracy','official']:command(['python','scripts/prepare_data.py'],'prepare-data',900)
    with (OUT/'launcher.log').open('w') as log:
        proc=subprocess.Popen(['bash','run.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state('loading',owner_pid=proc.pid)
        for _ in range(360):
            if proc.poll() is not None:raise RuntimeError(f'Service launcher exited {proc.returncode}')
            if (OUT/'ready.json').exists():break
            time.sleep(5)
        else:raise TimeoutError('Service readiness')
        state('functional');command(['bash','smoke.sh'],'smoke',1800)
        stages=['bench','accuracy','official'] if a.stage=='formal' else [a.stage]
        for stage in stages:
            state(stage);command(['bash',stage+'.sh'],stage,14400)
            if stage in ['accuracy','official']:command(['python','scripts/rescore.py','--stage',stage,'--output',str(OUT/stage)],stage+'-recheck',900)
            (ROOT/'results'/(stage+'-current.json')).write_text(json.dumps({'path':str(OUT),'output':str(OUT/stage),'controller_pid':os.getpid()}))
            print('STAGE_COMPLETE',stage,flush=True)
        command(['bash','stop.sh'],'stop',120);assert proc.wait(timeout=100)==0;proc=None
    state('completed',stages=stages,official_acceptance=None,independent_service_full_reproduction=None)
except BaseException as e:state('failed',error=repr(e));raise
finally:
    if proc is not None and proc.poll() is None:
        command(['bash','stop.sh'],'failure-cleanup',120,check=False)
        try:proc.wait(timeout=100)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=70)
