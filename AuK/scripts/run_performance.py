"""Run identical functional prefixes and workloads in two owned server processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import requests

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=True)
state = {'attention_backend': os.environ.get('AUK_ATTENTION_BACKEND','math'),
         'dit_compute_dtype':os.environ.get('AUK_DIT_DTYPE','float32'),
         'conditioning_compute_dtype':'bfloat16', 'rounds': [], 'complete': False,
         'byte_exact_waveform_repeatability_required_for_performance': False}

def save(phase, **extra):
    state.update(phase=phase, **extra)
    (OUT/'state.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({'phase':phase, **extra}), flush=True)

def stop(server):
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.getpgid(server.pid) == server.pid:
            os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)

try:
    for label in ['first', 'repeat']:
        folder = OUT/label
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, AUK_ATTENTION_BACKEND=state['attention_backend'], AUK_DIT_DTYPE=state['dit_compute_dtype'])
        with (folder/'server.log').open('wb') as log:
            server = subprocess.Popen(['bash', str(ROOT/'run.sh')], cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT,
                                      env=env, start_new_session=True)
            try:
                save(label+'_startup', server_pid=server.pid)
                deadline = time.monotonic()+360
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Owned server exited during startup')
                    try:
                        response = requests.get('http://127.0.0.1:30005/health', timeout=2)
                        if response.ok and response.json().get('running'):
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Server readiness timeout')
                    time.sleep(2)
                save(label+'_functional')
                with (folder/'smoke.log').open('wb') as client_log:
                    smoke = subprocess.run(['bash', str(ROOT/'run-scnet.sh'), sys.executable,
                           'scripts/http_smoke.py', '--output', str(folder/'smoke')], cwd=ROOT,
                           stdout=client_log, stderr=subprocess.STDOUT, timeout=600)
                summary = json.loads((folder/'smoke/summary.json').read_text())
                functional = len(summary['cases']) == 9 and all(r['passed'] for r in summary['cases'])
                if not functional:
                    raise RuntimeError('HTTP functional cases failed')
                save(label+'_performance', smoke_exit=smoke.returncode,
                     http_functionality_passed=functional,
                     byte_exact_waveform_repeatability=summary['checks']['same_seed_pcm_equal'])
                with (folder/'bench.log').open('wb') as client_log:
                    subprocess.run(['bash', str(ROOT/'run-scnet.sh'), sys.executable,
                           'scripts/http_bench.py', '--output', str(folder/'bench')], cwd=ROOT,
                           stdout=client_log, stderr=subprocess.STDOUT, timeout=1800, check=True)
                bench = json.loads((folder/'bench/summary.json').read_text())
                assert bench['complete']
                state['rounds'].append({'label':label, 'server_pid':server.pid,
                    'http_functionality_passed':functional,
                    'byte_exact_waveform_repeatability':summary['checks']['same_seed_pcm_equal'],
                    'performance':bench})
            finally:
                stop(server)
        save(label+'_stopped')
    differences = []
    for a,b in zip(state['rounds'][0]['performance']['cells'],state['rounds'][1]['performance']['cells']):
        assert (a['requested_seconds'],a['concurrency']) == (b['requested_seconds'],b['concurrency'])
        differences.append({'requested_seconds':a['requested_seconds'],'concurrency':a['concurrency'],
            **{key:abs(b[key]-a[key])/a[key]*100 for key in ['mean_latency_seconds','audio_seconds_per_wall_second']}})
    maximum = max(r[k] for r in differences for k in ['mean_latency_seconds','audio_seconds_per_wall_second'])
    save('finished', complete=True, differences_percent=differences,
         maximum_difference_percent=maximum, performance_reproduction_passed=maximum<=20)
except Exception as error:
    save('failed', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    raise
finally:
    with (OUT/'backup.log').open('wb') as log:
        subprocess.run([sys.executable, str(ROOT.parents[1]/'platform/scnet-notebook/backup_shared.py')],
                       stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=True)
