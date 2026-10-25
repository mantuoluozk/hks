"""Repeat identical action prompts before/after prefix-cache reuse."""
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = 'http://127.0.0.1:' + os.environ.get('PORT', '30125')
out = Path(os.environ.get('DIAGNOSTIC_OUTPUT_DIR', str(root / 'results' / ('cache-diagnostic-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))))
out.mkdir()
rows = []
for label in ['grounding', 'pointing', 'trajectory']:
    with opener.open(base + '/flush_cache', timeout=30) as response:
        assert response.status == 200
    source = Path(os.environ.get('REFERENCE_SMOKE_DIR', str(root / 'results/smoke-20261024T123424Z')))
    body = json.loads((source / f'{label}.request.json').read_text())
    results = []
    for trial in range(3):
        request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with opener.open(request, timeout=300) as response:
            data = json.load(response)
        (out / f'{label}-{trial}.json').write_text(json.dumps(data, indent=2))
        row = {'case': label, 'trial': trial, 'answer': data['choices'][0]['message']['content'], 'usage': data.get('usage')}
        results.append(row); print(json.dumps(row), flush=True)
    rows.append({'case': label, 'unique_answers': len({r['answer'] for r in results}), 'trials': results})
(out / 'summary.json').write_text(json.dumps({'purpose': 'cache diagnostic, not official evaluation', 'cases': rows}, indent=2))
