"""Fixed speech HTTP workload; end-to-end latency and generated audio throughput."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
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
cells = []

def send(duration, concurrency, index, warmup):
    name = f's{duration}-c{concurrency}-'+('warm' if warmup else 'measure')+str(index)
    body = {'input': 'Hello, welcome to our speech generation test.', 'seed': 1234+index,
            'response_format': 'wav', 'stage_params': {'auk_engine': {'gen_seconds': duration}}}
    started = time.perf_counter()
    response = requests.post(a.url+'/v1/audio/speech', json=body, timeout=300)
    latency = time.perf_counter()-started
    response.raise_for_status()
    raw = response.content
    waveform, rate = sf.read(io.BytesIO(raw), dtype='float32')
    assert rate == 24000 and waveform.ndim == 1 and np.isfinite(waveform).all()
    assert float(np.std(waveform)) > 1e-6 and abs(waveform.size/rate-duration) < .021
    (out/(name+'.wav')).write_bytes(raw)
    row = {'name': name, 'requested_seconds': duration, 'concurrency': concurrency,
           'warmup': warmup, 'seed': body['seed'], 'status': response.status_code,
           'latency_seconds': latency, 'audio_seconds': waveform.size/rate,
           'sample_rate': rate, 'samples': waveform.size,
           'response_sha256': hashlib.sha256(raw).hexdigest()}
    (out/(name+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    return row

for duration in [3, 5]:
    for concurrency in [1, 2]:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            warm = list(executor.map(lambda i: send(duration, concurrency, i, True), range(concurrency)))
            started = time.perf_counter()
            rows = list(executor.map(lambda i: send(duration, concurrency, i, False), range(8)))
            elapsed = time.perf_counter()-started
        latencies = [r['latency_seconds'] for r in rows]
        cell = {'requested_seconds': duration, 'concurrency': concurrency, 'requests': len(rows),
                'warmup_requests': len(warm), 'wall_seconds': elapsed,
                'mean_latency_seconds': float(np.mean(latencies)),
                'p95_latency_seconds': float(np.percentile(latencies,95)),
                'mean_rtf': float(np.mean(latencies))/duration,
                'audio_seconds_per_wall_second': sum(r['audio_seconds'] for r in rows)/elapsed}
        cells.append(cell)
        (out/'summary.json').write_text(json.dumps({'cells': cells, 'complete': len(cells)==4,
           'nfe':32, 'cfg_strength':2.0, 'sway':-1.0, 'server_max_batch_size':1,
           'attention_backend':os.environ.get('AUK_ATTENTION_BACKEND'),
           'dit_compute_dtype':os.environ.get('AUK_DIT_DTYPE'),
           'warmups_counted_in_metrics':False}, indent=2)+'\n')
        print(json.dumps(cell), flush=True)
