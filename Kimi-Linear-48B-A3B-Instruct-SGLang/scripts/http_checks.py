"""Check an explicitly started Kimi-Linear HTTP service; never start or stop it."""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--url', default='http://127.0.0.1:30148')
p.add_argument('--model', default='kimi-linear-48b-a3b')
p.add_argument('--output', type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
rows = []


def request(messages, **extra):
    body = {'model': a.model, 'messages': messages, 'temperature': 0, 'seed': 42,
            'max_tokens': 256, **extra}
    start = time.monotonic()
    req = urllib.request.Request(a.url + '/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            raw, status = response.read().decode(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read().decode(), error.code
    return {'request': body, 'status': status, 'raw': raw, 'seconds': time.monotonic() - start}


def content(row):
    return json.loads(row['raw'])['choices'][0]['message']['content'] or ''


def check(name, messages, expected):
    row = request(messages)
    row.update(name=name, passed=row['status'] == 200 and expected in content(row))
    rows.append(row)
    return row


check('english', [{'role': 'user', 'content': 'Reply with the word READY only.'}], 'READY')
check('chinese', [{'role': 'user', 'content': '计算2+3，只输出阿拉伯数字。'}], '5')
check('multi_turn', [{'role': 'user', 'content': 'Remember this code: 9381.'},
                     {'role': 'assistant', 'content': 'I will remember the code.'},
                     {'role': 'user', 'content': 'What is the code? Reply with the code only.'}], '9381')
row = request([{'role': 'user', 'content': 'Reply with READY only.'}], stream=True)
events = [line[6:] for line in row['raw'].splitlines() if line.startswith('data: ')]
text = ''.join(json.loads(s)['choices'][0].get('delta', {}).get('content') or ''
               for s in events if s != '[DONE]' and json.loads(s).get('choices'))
row.update(name='stream', passed=row['status'] == 200 and '[DONE]' in events and 'READY' in text)
rows.append(row)
row = request([{'role': 'user', 'content': 'Hello'}], max_tokens=-1)
row.update(name='invalid_parameter', passed=row['status'] == 400)
rows.append(row)
messages = [{'role': 'user', 'content': 'Reply with exactly: BLUE'}]
serial = request(messages)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    parallel = list(pool.map(lambda _: request(messages), range(2)))
rows.append({'name': 'serial_vs_concurrent', 'serial': serial, 'parallel': parallel,
             'passed': serial['status'] == 200 and bool(content(serial)) and
             all(r['status'] == 200 and content(r) == content(serial) for r in parallel)})
job = a.output or root / 'results' / ('smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
job.mkdir(parents=True, exist_ok=True)
result = {'checks': rows, 'passed': all(r['passed'] for r in rows), 'total': len(rows),
          'successes': sum(r['passed'] for r in rows)}
(job / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'job': str(job), 'passed': result['passed'], 'successes': result['successes'], 'total': len(rows)}))
raise SystemExit(0 if result['passed'] else 2)
