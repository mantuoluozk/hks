"""Fixed synthetic text workloads; record actual token usage and all responses."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--url', default='http://127.0.0.1:30148')
p.add_argument('--model', default='kimi-linear-48b-a3b')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rows = []


def request(repeats):
    text = 'The sky is blue. ' * repeats + '\nContinue with numbered short facts about science.'
    body = {'model': a.model, 'messages': [{'role': 'user', 'content': text}],
            'temperature': 0, 'seed': 42, 'max_tokens': 128, 'ignore_eos': True}
    start = time.perf_counter()
    req = urllib.request.Request(a.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    seconds = time.perf_counter() - start
    assert result['usage']['completion_tokens'] == 128, result
    assert result['choices'][0]['message']['content']
    return {'seconds': seconds, 'response': result, 'prompt_repeats': repeats}


for repeats in (32, 256):
    for concurrency in (1, 2, 4):
        for _ in range(3):
            row = request(repeats)
            row.update(concurrency=concurrency, warmup=True)
            rows.append(row)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            start = time.perf_counter()
            values = list(pool.map(lambda _: request(repeats), range(3 * concurrency)))
            wall = time.perf_counter() - start
        for row in values:
            row.update(concurrency=concurrency, warmup=False, batch_wall_seconds=wall)
            rows.append(row)
        with (a.output / 'requests.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
print(json.dumps({'requests': len(rows), 'timed_requests': sum(not r['warmup'] for r in rows)}))
