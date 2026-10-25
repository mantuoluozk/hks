"""Compare concurrent action responses with same-service serial smoke outputs."""
import concurrent.futures as cf
import hashlib
import json
import os
import urllib.request
from pathlib import Path
root = Path(__file__).resolve().parents[1]
source = Path(os.environ['REFERENCE_SMOKE_DIR'])
out = Path(os.environ['CONCURRENCY_OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=False)
summary = json.loads((source / 'summary.json').read_text())
expected = {c['case']: c['answer'] for c in summary['cases']}
base = 'http://127.0.0.1:' + os.environ['PORT']
def run(task):
    label, trial = task
    raw = (source / f'{label}.request.json').read_bytes()
    body = json.loads(raw)
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type':'application/json'})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=600) as response:
            data = json.load(response)
        answer = data['choices'][0]['message']['content']
        (out / f'{label}-{trial}.response.json').write_text(json.dumps(data, indent=2))
        return {'case': label, 'trial': trial, 'input_sha256': hashlib.sha256(raw).hexdigest(),
                'answer': answer, 'exact_match': answer == expected[label], 'error': None}
    except Exception as exc:
        return {'case': label, 'trial': trial, 'exact_match': False, 'error': type(exc).__name__}
tasks = [(label, trial) for trial in range(8) for label in ['grounding','pointing','trajectory']]
with cf.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(run, tasks))
record = {'purpose': 'local concurrent versus serial response consistency, not official GPU accuracy',
          'workers':4,'requests':len(rows),'exact_matches':sum(r['exact_match'] for r in rows),
          'errors':sum(r['error'] is not None for r in rows),'cases':rows,
          'official_accuracy_pass':None,'official_scnet_validation':False}
(out / 'summary.json').write_text(json.dumps(record, indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='cases'}, indent=2))
assert record['requests']==record['exact_matches']==24 and record['errors']==0
