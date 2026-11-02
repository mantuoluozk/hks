"""Compare eager and decode graphs using the unchanged historical short matrix."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument('--resume-baseline', help='Existing job whose first baseline is still running')
parser.add_argument('--repeats', type=int, choices=(1, 2), default=1)
args = parser.parse_args()
OUT = ROOT / 'results' / (Path(args.resume_baseline).name if args.resume_baseline else time.strftime('optimization-%Y%m%dT%H%M%SZ', time.gmtime()))
OUT.mkdir(exist_ok=bool(args.resume_baseline))
(OUT / 'source').mkdir(exist_ok=True)
for rel in ['scripts/run_optimization_screen.py', 'scripts/bench_http.py',
            'scripts/smoke_http.py', 'serve-scnet.sh', 'run-scnet.sh']:
    dest=OUT / 'source' / Path(rel).name
    if not dest.exists():dest.write_bytes((ROOT / rel).read_bytes())
(OUT / 'source' / 'active_controller.py').write_bytes(Path(__file__).read_bytes())
(ROOT / 'results/optimization-current.json').write_text(json.dumps({'path': str(OUT), 'pid': os.getpid()}))
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def state(**kw):
    (OUT / 'state.json').write_text(json.dumps({'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw}, indent=2))

def run(args, log, timeout):
    with log.open('w') as f:
        subprocess.run(['bash', 'run-scnet.sh'] + args, stdout=f, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)

with socket.socket() as s:
    if s.connect_ex(('127.0.0.1', 30004)) == 0 and not args.resume_baseline:
        raise RuntimeError('Port already occupied')

proc = None
try:
    if args.resume_baseline:
        state(status='waiting_existing_baseline', repeats=args.repeats)
        for _ in range(180):
            if (OUT / 'baseline/bench1/summary.json').exists():break
            time.sleep(5)
        else:raise TimeoutError('Existing baseline did not finish')
        pid=int((OUT / 'baseline/server.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'vllm.entrypoints.openai.api_server' in cmd
        os.kill(pid,signal.SIGTERM)
        for _ in range(90):
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',30004))!=0:break
            time.sleep(1)
        else:raise TimeoutError('Baseline service did not stop')
        time.sleep(5)
    for name in (['decode-graph'] if args.resume_baseline else ['baseline', 'decode-graph']):
        target = OUT / name
        target.mkdir()
        launcher = (ROOT / 'serve-scnet.sh').read_text()
        if name == 'decode-graph':
            config = json.dumps({'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                                 'cudagraph_capture_sizes': [1, 2]}, separators=(',', ':'))
            launcher = launcher.replace('--enforce-eager', "--compilation-config '" + config + "'")
        (target / 'launcher.sh').write_text(launcher)
        # Preserve launcher cwd semantics: execute text from model root.
        with (target / 'server.log').open('w') as log:
            proc = subprocess.Popen(['bash', '-c', launcher.replace('cd -- "$(dirname -- "${BASH_SOURCE[0]}")"', '')],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (target / 'server.pid').write_text(str(proc.pid))
            state(status='loading', candidate=name, server_pid=proc.pid)
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError(name + ': server exited')
                try:
                    with op.open('http://127.0.0.1:30004/health', timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(5)
            else:
                raise TimeoutError('Readiness timeout')
            state(status='functional', candidate=name)
            run(['python', 'scripts/smoke_http.py', '--output', str(target / 'functional')], target / 'smoke.log', 600)
            for repeat in range(1,args.repeats+1):
                state(status='benchmarking', candidate=name, repeat=repeat)
                run(['python', 'scripts/bench_http.py', '--smoke-summary', str(target / 'functional/summary.json'),
                     '--output', str(target / f'bench{repeat}')], target / f'bench{repeat}.log', 900)
            proc.terminate()
            proc.wait(timeout=90)
            proc = None
        time.sleep(5)
    comparison = []
    for repeat in range(1,args.repeats+1):
        a = json.loads((OUT / f'baseline/bench{repeat}/summary.json').read_text())['workloads']
        b = json.loads((OUT / f'decode-graph/bench{repeat}/summary.json').read_text())['workloads']
        for x, y in zip(a, b):
            assert (x['concurrency'], x['max_tokens']) == (y['concurrency'], y['max_tokens'])
            comparison.append({'repeat': repeat, 'concurrency': x['concurrency'], 'max_tokens': x['max_tokens'],
                               'baseline_tps': x['output_tokens_per_second'], 'candidate_tps': y['output_tokens_per_second'],
                               'throughput_gain_pct': (y['output_tokens_per_second'] / x['output_tokens_per_second'] - 1) * 100,
                               'same_output_counts': x['completion_tokens'] == y['completion_tokens']})
    (OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    state(status='completed', official_acceptance=None)
except Exception as exc:
    state(status='failed', error=repr(exc))
    raise
finally:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
