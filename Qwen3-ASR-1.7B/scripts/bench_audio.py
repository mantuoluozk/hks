"""Measure real-audio API latency/RTF; use natural EOS, not forced token counts."""
import base64
import concurrent.futures as cf
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import soundfile as sf

root = Path(__file__).resolve().parents[1]
runtime = json.loads((Path(os.environ.get('RUN_RESULTS_DIR', str(root/'results')))/'runtime-config.json').read_text())
out = Path(os.environ.get('BENCH_OUTPUT_DIR', str(root/'bench'/('audio-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))))
out.mkdir(parents=True, exist_ok=False)
payloads = []
for lang in ['en', 'zh']:
    path = root/f'data/examples/asr_{lang}.wav'
    payloads.append({'language': lang, 'seconds': sf.info(path).duration,
                     'url': 'data:audio/wav;base64,'+base64.b64encode(path.read_bytes()).decode()})
url = f'http://127.0.0.1:{runtime["PORT"]}/v1/chat/completions'

def infer(item):
    session = requests.Session();session.trust_env = False
    t0 = time.perf_counter()
    r = session.post(url, json={'model': runtime['SERVED_NAME'], 'temperature': 0,
        'max_tokens': 1024, 'messages': [{'role': 'user', 'content': [
            {'type': 'audio_url', 'audio_url': {'url': item['url']}}]}]}, timeout=300)
    r.raise_for_status();response = r.json()
    choice = response['choices'][0]
    assert choice['finish_reason'] != 'length', 'Audio benchmark truncated'
    text = choice['message']['content'];assert text.split('<asr_text>',1)[-1].strip()
    elapsed = time.perf_counter()-t0
    return {'language_input': item['language'], 'audio_seconds': item['seconds'],
            'elapsed_seconds': elapsed, 'rtf': elapsed/item['seconds'], 'output': text,
            'completion_tokens': response.get('usage', {}).get('completion_tokens')}

results = []
with (out/'requests.jsonl').open('w', encoding='utf-8') as detail:
    for concurrency in [1, 4, 8, 16]:
        count = max(8, 2*concurrency)
        for round_id in [1, 2]:
            # Same fixtures and decoding in each profile; prefix/MM caches disabled at startup.
            for payload in payloads:infer(payload)
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(pool.map(infer, [payloads[i % 2] for i in range(count)]))
            wall = time.perf_counter()-t0
            for i, row in enumerate(rows):
                detail.write(json.dumps({'round': round_id, 'concurrency': concurrency, 'request': i, **row}, ensure_ascii=False)+'\n')
            detail.flush()
            total_audio = sum(r['audio_seconds'] for r in rows)
            item = {'round': round_id, 'concurrency': concurrency, 'requests': count,
                    'wall_seconds': wall, 'audio_seconds': total_audio,
                    'audio_seconds_per_wall_second': total_audio/wall,
                    'mean_request_rtf': statistics.mean(r['rtf'] for r in rows),
                    'mean_latency_seconds': statistics.mean(r['elapsed_seconds'] for r in rows),
                    'max_latency_seconds': max(r['elapsed_seconds'] for r in rows)}
            results.append(item);print(json.dumps(item), flush=True)
report = {'runtime': runtime, 'fixture_source': 'Two official demonstration WAV files, 48kHz English and 16kHz Chinese',
          'workloads': results, 'counted_requests': sum(r['requests'] for r in results),
          'round_reported': 2, 'prefix_caching': False, 'mm_processor_cache_gb': 0,
          'decode': 'BF16, greedy, max_tokens=1024, natural EOS, language not specified',
          'official_performance_protocol': False, 'official_accuracy_pass': None}
(out/'summary.json').write_text(json.dumps(report, indent=2))
print('Completed', len(results), 'audio workloads:', out, flush=True)
