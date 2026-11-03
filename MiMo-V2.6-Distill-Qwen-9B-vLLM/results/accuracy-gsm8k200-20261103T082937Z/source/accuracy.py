"""One owned final-configuration service for SGLang-aligned 200 GSM8K, then stop."""
import json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'results'/time.strftime('accuracy-gsm8k200-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir()
(ROOT/'results/accuracy-current.json').write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
(OUT/'source').mkdir()
for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.sh')):shutil.copyfile(p,OUT/'source'/p.name)
env=dict(os.environ,EXECUTION_MODE='graph',RUN_RESULTS_DIR=str(OUT),SMOKE_OUTPUT_DIR=str(OUT/'functional'))
def state(stage,**extra):(OUT/'state.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
def command(cmd,name,timeout):
    with (OUT/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
proc=None
try:
    with (OUT/'launcher.log').open('w') as log:
        proc=subprocess.Popen(['bash','run.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state('loading',owner_pid=proc.pid)
        for _ in range(360):
            if proc.poll() is not None:raise RuntimeError(f'Accuracy service exited {proc.returncode}')
            if (OUT/'ready.json').exists():break
            time.sleep(5)
        else:raise TimeoutError('Accuracy service readiness')
        state('functional');command(['bash','smoke.sh'],'smoke',1800)
        state('gsm8k',total=200,workers=8)
        command(['python','scripts/run_gsm8k.py','--output',str(OUT/'gsm8k')],'gsm8k',43200)
        state('rescoring');command(['python','scripts/rescore_gsm8k.py','--output',str(OUT/'gsm8k')],'rescore',300)
        command(['bash','stop.sh'],'stop',100);assert proc.wait(timeout=90)==0;proc=None
    state('completed',dataset='GSM8K fixed SGLang-aligned sample',total=200,official_acceptance=None)
except BaseException as e:state('failed',error=repr(e));raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=100)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
