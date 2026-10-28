"""Exercise speech, editing, repeatability, and validation over real HTTP."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import requests
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--url', default='http://127.0.0.1:30005')
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
rows = []
audio = {}

def call(name, body, endpoint='/v1/audio/speech', negative=False):
    (out/(name+'-request.json')).write_text(json.dumps(body, ensure_ascii=False, indent=2)+'\n')
    started = time.perf_counter()
    row = {'name': name, 'endpoint': endpoint, 'passed': False}
    try:
        response = requests.post(a.url+endpoint, json=body, timeout=240)
        row.update(status=response.status_code, seconds=time.perf_counter()-started,
                   content_type=response.headers.get('content-type'))
        (out/(name+'-response.bin')).write_bytes(response.content)
        row['response_sha256'] = hashlib.sha256(response.content).hexdigest()
        if negative:
            row['passed'] = 400 <= response.status_code < 500
        else:
            response.raise_for_status()
            raw = response.content
            if endpoint == '/generate':
                raw = base64.b64decode(response.json()['audio']['data'])
            wave, rate = sf.read(io.BytesIO(raw), dtype='float32')
            assert wave.ndim == 1 and rate == 24000
            assert np.isfinite(wave).all() and float(np.std(wave)) > 1e-6
            assert abs(wave.size/rate-4) < .021
            audio[name] = wave
            (out/(name+'.wav')).write_bytes(raw)
            row.update(sample_rate=rate, samples=wave.size, rms=float(np.sqrt(np.mean(wave**2))),
                       waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(), passed=True)
    except Exception as error:
        row['error'] = type(error).__name__+': '+str(error)
    rows.append(row)
    print(json.dumps(row), flush=True)

body = {'input': 'Hello, welcome to our speech generation test.', 'seed': 42,
        'response_format': 'wav', 'stage_params': {'auk_engine': {'gen_seconds': 4}}}
health = requests.get(a.url+'/health', timeout=10)
(out/'health.json').write_text(health.text)
assert health.status_code == 200 and health.json()['running']
call('english', body)
call('chinese', dict(body, input='你好，欢迎参加语音生成测试。'))
call('repeat_seed42', body)
call('different_seed43', dict(body, seed=43))
call('voice_instruction', dict(body, instructions='A calm, warm female voice.'))
if 'english' in audio:
    reference = 'data:audio/wav;base64,'+base64.b64encode((out/'english.wav').read_bytes()).decode()
    call('reference_voice', dict(body, ref_audio=reference))
    call('editing', {'prompt': 'Remove the background noise.',
                     'metadata': {'tts_params': {'ref_audio': reference}},
                     'sampling_params': {'seed': 42}, 'output_modalities': ['audio'],
                     'return_logprob': False}, endpoint='/generate')
call('empty_input', dict(body, input=''), negative=True)
call('zero_duration', dict(body, stage_params={'auk_engine': {'gen_seconds': 0}}), negative=True)
checks = {}
if {'english', 'repeat_seed42'} <= audio.keys():
    checks['same_seed_pcm_equal'] = np.array_equal(audio['english'], audio['repeat_seed42'])
    checks['same_seed_max_abs_difference'] = float(np.max(np.abs(audio['english']-audio['repeat_seed42'])))
if {'english', 'different_seed43'} <= audio.keys():
    checks['different_seed_pcm_differs'] = not np.array_equal(audio['english'], audio['different_seed43'])
passed = (len(rows) == 9 and all(row['passed'] for row in rows)
          and checks.get('same_seed_pcm_equal', False) and checks.get('different_seed_pcm_differs', False))
result = {'cases': rows, 'checks': checks, 'passed': passed,
          'semantic_quality_measured': False, 'official_accuracy_tested': False}
(out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'passed': passed, 'checks': checks}), flush=True)
raise SystemExit(0 if passed else 2)
