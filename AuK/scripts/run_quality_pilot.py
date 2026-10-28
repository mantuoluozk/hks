"""Sequentially generate and transcribe eight examples; never overlap GPU evaluators."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import requests

p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--max-samples',type=int,default=8,help='0 selects the full 1088-row English split')
p.add_argument('--seed',type=int,default=1234)
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def state(phase,**kwargs):
    d={'phase':phase,**kwargs};(out/'state.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
def run(script,*args,log):
    with (out/log).open('wb') as f:
        subprocess.run(['bash',str(ROOT/'run-scnet.sh'),sys.executable,'scripts/'+script,*args],
                       cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
assert json.loads((ROOT/'results/performance-fp32-20261028/state.json').read_text())['performance_reproduction_passed']
server=None
try:
    with (out/'server.log').open('wb') as f:
        server=subprocess.Popen(['bash',str(ROOT/'run.sh')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state('server_startup',server_pid=server.pid)
        deadline=time.monotonic()+360
        while True:
            if server.poll() is not None:raise RuntimeError('Pilot server exited')
            try:
                if requests.get('http://127.0.0.1:30005/health',timeout=2).json().get('running'):break
            except (requests.RequestException,ValueError):pass
            if time.monotonic()>deadline:raise TimeoutError('Pilot startup timeout')
            time.sleep(2)
        try:
            state('generating',selected_rows=a.max_samples or 1088,seed=a.seed)
            generation_args=['--output',str(out/'generation'),'--seed',str(a.seed)]
            if a.max_samples>0:generation_args.extend(['--max-samples',str(a.max_samples)])
            with (out/'generation.log').open('wb') as f:
                subprocess.run(['bash',str(ROOT/'run-scnet.sh'),sys.executable,'scripts/generate_seedtts.py',*generation_args],
                               cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=21600)
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.getpgid(server.pid)==server.pid:os.killpg(server.pid,signal.SIGKILL)
                    server.wait(timeout=10)
    state('transcribing',selected_rows=a.max_samples or 1088)
    with (out/'score.log').open('wb') as f:
        subprocess.run(['bash',str(ROOT/'run-scnet.sh'),sys.executable,'scripts/score_seedtts.py','--generation',str(out/'generation')],
                       cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=21600)
    summary=json.loads((out/'generation/wer-summary.json').read_text())
    state('finished',accuracy_summary=summary,full_dataset_evaluated=a.max_samples<=0)
except Exception as error:
    state('failed',error=type(error).__name__+': '+str(error));raise
finally:
    if server is not None and server.poll() is None:
        server.terminate();server.wait(timeout=30)
    with (out/'backup.log').open('wb') as f:
        subprocess.run([sys.executable,str(ROOT.parents[1]/'platform/scnet-notebook/backup_shared.py')],
                       stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
