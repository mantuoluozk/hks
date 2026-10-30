"""OCRBench only: 16x instruct, complete 1000 questions, independent fresh run."""
import fcntl
import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from eval_accuracy_v2 import ROOT, run

parser = argparse.ArgumentParser()
parser.add_argument('--prior-server-pid', type=int)
options = parser.parse_args()
lock = (ROOT / 'results/ocrbench-review.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
tag = 'ocrbench-16x-review-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
job = ROOT / 'results/accuracy' / tag
raw = ROOT / 'data/accuracy-runs' / tag
job.mkdir(parents=True)
raw.mkdir(parents=True)
(ROOT / 'results/accuracy-current.json').write_text(json.dumps({'job': str(job), 'raw': str(raw), 'controller_pid': os.getpid()}, indent=2) + '\n')


def state(stage, **kw):
    (job / 'state.json').write_text(json.dumps({'stage': stage, **kw}, indent=2) + '\n')


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGINT, interrupt)
source = job / 'source'
source.mkdir()
for p in (ROOT / 'scripts').glob('*.py'):
    shutil.copy2(p, source / p.name)
for p in ROOT.glob('*.sh'):
    shutil.copy2(p, source / p.name)
shutil.copytree(ROOT / 'python/minicpm_adapter', source / 'minicpm_adapter', ignore=shutil.ignore_patterns('__pycache__'))
hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
(job / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
(job / 'intent.json').write_text(json.dumps({'dataset': 'OCRBench', 'full_rows': 1000, 'official_16x_instruct_reference': 90.40,
    'thinking': False, 'visual_compression': '16x', 'max_slice_nums': 9, 'output_budget': 1024,
    'context': 8192, 'workers': 4, 'temperature': 0, 'seed': 42,
    'pilot_gate': 'transport only, no accuracy gate', 'other_datasets_cancelled': ['MMMU', 'ChartQA'],
    'official_full_protocol_unknown': True, 'legacy_score_preserved': 84.8}, indent=2) + '\n')
proc = None
state('preflight')
try:
    # Previous multi-benchmark controller was explicitly stopped by user steering.
    deadline = time.monotonic() + 120
    while options.prior_server_pid and Path(f'/proc/{options.prior_server_pid}').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Previous owned server has not exited')
        time.sleep(2)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', 30147)) == 0:
            raise RuntimeError('Port occupied; refusing to reuse another service')
    env = dict(os.environ, MAX_MODEL_LEN='8192', MAX_NUM_SEQS='4')
    log = (job / 'server.log').open('w')
    proc = subprocess.Popen(['bash', 'run.sh'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (job / 'owned-server.pid').write_text(str(proc.pid) + '\n')
    state('starting-server', owned_pid=proc.pid)
    deadline = time.monotonic() + 2400
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f'Owned service exited {proc.returncode}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:30147/health', timeout=3) as r:
                assert r.status == 200
            break
        except Exception:
            if time.monotonic() > deadline:
                raise TimeoutError('Service startup timed out')
            time.sleep(2)
    args = Path(f'/proc/{proc.pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert '--max-model-len 8192' in args and '--mamba-ssm-cache-dtype float32' in args
    (job / 'actual-server-command.json').write_text(json.dumps({'pid': proc.pid, 'command': args}, indent=2) + '\n')
    state('pilot')
    pilot = job / 'pilot'
    pilot.mkdir()
    shutil.copytree(source, pilot / 'source')
    items = json.loads((ROOT / 'data/OCRBench/data.json').read_text())
    indices, seen = [], set()
    for i, item in enumerate(items):
        if item['question_type'] not in seen:
            seen.add(item['question_type'])
            indices.append(i)
    indices = sorted(set(indices + [743]))
    (job / 'pilot-selection.json').write_text(json.dumps(indices) + '\n')
    pilot_summary = run('ocrbench', pilot, raw / 'pilot', 4, 'http://127.0.0.1:30147', indices, mode='16x')
    if pilot_summary['http_errors']:
        raise RuntimeError('OCRBench pilot transport/capacity failure')
    state('running-full', dataset='ocrbench')
    summary = run('ocrbench', job, raw / 'full', 4, 'http://127.0.0.1:30147', mode='16x')
    (job / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    state('completed')
except BaseException as e:
    state('failed', error=repr(e))
    raise
finally:
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    with (job / 'backup.log').open('w') as f:
        subprocess.run([sys.executable, str(ROOT.parents[1] / 'platform/scnet-notebook/backup_shared.py')], stdout=f, stderr=subprocess.STDOUT, check=True)
