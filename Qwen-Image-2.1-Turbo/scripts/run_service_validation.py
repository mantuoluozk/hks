"""Run functional cases then two independent SGLang service performance rounds."""
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import requests

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/scnet'/('service-validation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True)
state={'status':'running','rounds':[],'official_accuracy_pass':None,
       'protocol':'identical eight-case functional prefix and per-size warmup in both service processes'}


def save():
    (out/'status.json').write_text(json.dumps(state,indent=2))


def run_script(args,folder,env):
    folder.mkdir(parents=True)
    env=env.copy();env['RUN_RESULTS_DIR']=str(folder)
    with (folder/'command.log').open('w') as f:
        subprocess.run(['bash','run-scnet.sh','python',*args],cwd=ROOT,env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)


def stop_server(server):
    if server.poll() is not None:return
    os.killpg(server.pid,signal.SIGTERM)
    try:server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)


save()
try:
    for index in [1,2]:
        with socket.socket() as probe:
            probe.settimeout(2)
            if probe.connect_ex(('127.0.0.1',30003))==0:
                raise RuntimeError('Port 30003 already in use; do not test or stop an unrelated service')
        folder=out/f'round-{index}';folder.mkdir()
        env=os.environ.copy();env['HKS_TRACE_DIR']=str(folder/'trace')
        args=['bash','run-scnet.sh','python','scripts/serve_sglang.py',
              '--model-path',env.get('MODEL_PATH','/root/private_data/SothisAI/model/Aihub/Qwen-Image-2.1-Turbo/main/Qwen-Image-2.1-Turbo'),
              '--backend','diffusers','--num-gpus','1','--dit-precision','bf16',
              '--host','127.0.0.1','--port','30003']
        with (folder/'server.log').open('w') as f:
            server=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        row={'round':index,'pid':server.pid,'args':args,'status':'starting'}
        state['rounds'].append(row);save()
        try:
            s=requests.Session();s.trust_env=False
            for _ in range(180):
                if server.poll() is not None:raise RuntimeError('Service exited during startup')
                try:
                    if s.get('http://127.0.0.1:30003/health',timeout=3).status_code==200:break
                except requests.RequestException:pass
                time.sleep(5)
            else:raise TimeoutError('Service startup exceeded 15 minutes')
            row['status']='ready';save()
            run_script(['scripts/smoke_api.py'],folder/'smoke',env)
            cases=json.loads((folder/'smoke/summary.json').read_text())['cases']
            assert len(cases)==8
            row['functional_cases']=len(cases)
            row['status']='benchmarking';save()
            run_script(['scripts/benchmark_api.py','--output',str(folder/'benchmark'),'--repeats','3'],folder/'benchmark-process',env)
            row['status']='completed';save()
        finally:stop_server(server)
    first=json.loads((out/'round-1/benchmark/summary.json').read_text())['cells']
    second=json.loads((out/'round-2/benchmark/summary.json').read_text())['cells']
    comparisons=[{'size':a['size'],'first_mean_seconds':a['mean_seconds'],
                  'second_mean_seconds':b['mean_seconds'],
                  'relative_difference_percent':100*abs(b['mean_seconds']-a['mean_seconds'])/a['mean_seconds']}
                 for a,b in zip(first,second)]
    assert len(comparisons)==3 and all(a['size']==b['size'] for a,b in zip(first,second))
    state.update(status='completed',performance_comparison=comparisons)
except Exception:
    state.update(status='failed',error=traceback.format_exc());raise
finally:
    save()
    with (out/'backup.log').open('w') as f:
        result=subprocess.run(['python','/root/private_data/hks/platform/scnet-notebook/backup_shared.py'],
                              stdout=f,stderr=subprocess.STDOUT)
    state['backup_exit_code']=result.returncode;save()
    print(json.dumps({'results':str(out),'status':state['status']},indent=2),flush=True)
