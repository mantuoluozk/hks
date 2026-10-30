"""Generate the fixed 120 AuK base outputs; no quality scoring or performance ranking."""
import argparse
import base64
import csv
import fcntl
import hashlib
import importlib.metadata as md
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(__doc__)
p.add_argument('--conditions', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
run_lock = (ROOT / 'results/official-conditions.lock').open('a')
fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
lock = json.loads((ROOT / 'sources/official-t03t04-20261030.json').read_text())
assert hashlib.sha256(a.conditions.read_bytes()).hexdigest() == lock['condition_sha256']
cases = [json.loads(s) for s in a.conditions.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
assert len(cases) == len({c['id'] for c in cases}) == lock['rows'] == 120
for name, sha in lock['baseline_files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
assets = {}
for c in cases:
    assert c['model'] == 'auk' and c['seed'] in lock['seeds']
    assert c['params']['nfe'] == 32 and c['params']['cfg_strength'] == 2.0 and c['params']['sway_sampling_coef'] == -1.0
    assert c['text'] == c['input'] and c['text'].strip() and c['ref_text'].strip()
    asset = c['assets']['reference_audio']
    path = (a.conditions.parent / asset['path']).resolve(strict=True)
    if path not in assets:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == asset['sha256'], path
        assets[path] = base64.b64encode(raw).decode()
with socket.socket() as s:
    assert s.connect_ex(('127.0.0.1', 30005)) != 0, 'Refusing to reuse an existing service'
a.output.mkdir(parents=True, exist_ok=False)
(a.output / 'outputs').mkdir()
for name in ('scripts/run_official_conditions.py', *lock['baseline_files'], 'sources/official-t03t04-20261030.json'):
    target = a.output / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)


def write(name, value):
    (a.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def state(phase, **kwargs):
    write('state.json', {'phase': phase, **kwargs})
    print(json.dumps({'phase': phase, **kwargs}), flush=True)


def verify_files(meta_name, base, selected=None):
    meta = json.loads((ROOT / 'sources' / meta_name).read_text())
    records = []
    for item in meta['siblings']:
        name = item['rfilename']
        if selected is not None and name not in selected:
            continue
        if selected is None and Path(name).suffix not in {'.safetensors', '.json', '.txt'}:
            continue
        path = base / name
        assert path.stat().st_size == item['size'], path
        h = hashlib.sha256() if 'lfs' in item else hashlib.sha1(f'blob {item["size"]}\0'.encode())
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        expected = item['lfs']['sha256'] if 'lfs' in item else item['blobId']
        assert h.hexdigest() == expected, path
        records.append({'file': name, 'bytes': item['size'], 'hash': expected, 'passed': True})
    assert records
    return {'revision': meta['sha'], 'files': records}


server = None
def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    state('checking_weights', input_rows=120, reference_files=len(assets))
    model = Path(os.environ.get('AUK_MODEL_PATH', '/root/private_data/SothisAI/model/Aihub/AuK/main/AuK'))
    encoder = Path(os.environ.get('AUK_ENCODER_PATH', str(ROOT / 'weights/Qwen2.5-Omni-3B')))
    weights = verify_files('publisher-files.json', model, {'auk_base.safetensors', 'vae.safetensors', 'config.yaml', 'LICENSE'})
    conditioner = verify_files('conditioner-files.json', encoder)
    write('weights-check.json', {'model': weights, 'conditioner': conditioner})
    environment = {'sourcefind_commit': lock['sourcefind_commit'], 'model_revision': weights['revision'],
                   'encoder_revision': conditioner['revision'], 'condition_sha256': lock['condition_sha256'],
                   'packages': {n: md.version(n) for n in ('torch', 'sglang', 'transformers', 'openai')},
                   'framework': 'SGLang-Omni', 'framework_revision': '428b73225f808f0d93669140983e90839c9aea0e',
                   'device': torch.cuda.get_device_name(0), 'device_count_used': 1,
                   'conditioning_dtype': 'bfloat16', 'dit_dtype': 'float32', 'vae_dtype': 'float32',
                   'nfe': 32, 'cfg_strength': 2.0, 'sway_sampling_coef': -1.0, 'attention_backend': 'math',
                   'range': 'auk.jsonl, all 120 IDs', 'scores_computed': False}
    write('environment.json', environment)
    (a.output / '环境说明.txt').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
    with (a.output / 'server.log').open('w') as log:
        server = subprocess.Popen(['bash', str(ROOT / 'run.sh')], cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        (a.output / 'owned-server.pid').write_text(str(server.pid))
        state('starting_server', pid=server.pid)
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Owned server exited: {server.returncode}')
            try:
                if requests.get('http://127.0.0.1:30005/health', timeout=2).json().get('running'):
                    break
            except (requests.RequestException, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Server startup timeout')
            time.sleep(2)
        rows = []
        with (a.output / '反馈表.csv').open('w', encoding='utf-8-sig', newline='') as feedback:
            writer = csv.DictWriter(feedback, fieldnames=['condition_id', 'model', 'status', 'output_path', 'error', 'parameter_changes'])
            writer.writeheader()
            for c in cases:
                row = {'condition_id': c['id'], 'model': 'auk', 'status': '失败', 'output_path': '', 'error': '', 'parameter_changes': '无'}
                record = {'id': c['id'], 'seed': c['seed'], 'params': c['params'], 'passed': False}
                try:
                    path = (a.conditions.parent / c['assets']['reference_audio']['path']).resolve()
                    body = {'input': c['text'], 'ref_text': c['ref_text'], 'ref_audio': 'data:audio/wav;base64,' + assets[path],
                            'instructions': c['instructions'], 'seed': c['seed'], 'response_format': 'wav',
                            'stage_params': {'auk_engine': {'gen_seconds': c['params']['gen_seconds']}}}
                    response = requests.post('http://127.0.0.1:30005/v1/audio/speech', json=body, timeout=300)
                    response.raise_for_status()
                    output = a.output / 'outputs' / (c['id'] + '.wav')
                    output.write_bytes(response.content)
                    audio, rate = sf.read(io.BytesIO(response.content), dtype='float32')
                    assert rate == 24000 and audio.ndim == 1 and np.isfinite(audio).all() and float(np.std(audio)) > 1e-6
                    duration = len(audio) / rate
                    assert c['expected']['duration_min_s'] <= duration <= c['expected']['duration_max_s'], duration
                    record.update(passed=True, sample_rate=rate, channels=1, samples=len(audio), duration_seconds=duration,
                                  sha256=hashlib.sha256(response.content).hexdigest())
                    row.update(status='成功', output_path='outputs/' + output.name)
                except Exception as error:
                    row['error'] = type(error).__name__ + ': ' + str(error)
                rows.append(record)
                writer.writerow(row)
                feedback.flush()
                with (a.output / 'records.jsonl').open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                state('generating', completed=len(rows), successful=sum(r['passed'] for r in rows), total=120)
        write('summary.json', {'expected': 120, 'completed': len(rows), 'successful': sum(r['passed'] for r in rows),
                               'all_passed': all(r['passed'] for r in rows), 'scores_computed': False})
        state('completed', successful=sum(r['passed'] for r in rows), total=120)
except BaseException as error:
    state('failed', error=repr(error))
    raise
finally:
    if server is not None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)
    with (a.output / 'backup.log').open('w') as f:
        subprocess.run(['python', str(ROOT.parents[1] / 'platform/scnet-notebook/backup_shared.py')],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
