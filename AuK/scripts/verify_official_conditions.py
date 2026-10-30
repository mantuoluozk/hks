"""Independently check the official AuK output bundle, without running the model."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

p = argparse.ArgumentParser(__doc__)
p.add_argument('--job', type=Path, required=True)
p.add_argument('--conditions', type=Path, required=True)
a = p.parse_args()
conditions = [json.loads(s) for s in a.conditions.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
expected = {c['id']: c for c in conditions}
assert len(expected) == len(conditions) == 120
environment = json.loads((a.job / 'environment.json').read_text())
assert hashlib.sha256(a.conditions.read_bytes()).hexdigest() == environment['condition_sha256']
with (a.job / '反馈表.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == len({r['condition_id'] for r in rows}) == 120
assert {r['condition_id'] for r in rows} == set(expected)
records = [json.loads(s) for s in (a.job / 'records.jsonl').read_text().splitlines()]
assert len(records) == len({r['id'] for r in records}) == 120
by_id = {r['id']: r for r in records}
manifest = []
for row in rows:
    uid = row['condition_id']
    c, record = expected[uid], by_id[uid]
    assert row['status'] == '成功' and not row['error'] and row['parameter_changes'] == '无', uid
    assert row['model'] == 'auk' and row['output_path'] == 'outputs/' + uid + '.wav', uid
    assert record['passed'] and record['seed'] == c['seed'] and record['params'] == c['params'], uid
    path = a.job / row['output_path']
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == record['sha256'], uid
    audio, rate = sf.read(path, dtype='float32')
    assert rate == 24000 and audio.ndim == 1 and np.isfinite(audio).all() and float(np.std(audio)) > 1e-6, uid
    duration = len(audio) / rate
    assert c['expected']['duration_min_s'] <= duration <= c['expected']['duration_max_s'], uid
    assert record['samples'] == len(audio) and record['duration_seconds'] == duration, uid
    manifest.append({'id': uid, 'language': c['language'], 'seed': c['seed'], 'path': row['output_path'],
                     'bytes': len(raw), 'sha256': digest, 'sample_rate': rate, 'channels': 1,
                     'samples': len(audio), 'duration_seconds': duration})
summary = json.loads((a.job / 'summary.json').read_text())
assert summary['expected'] == summary['completed'] == summary['successful'] == 120 and summary['all_passed']
result = {'expected': 120, 'verified_outputs': len(manifest), 'failed': 0,
          'language_counts': dict(collections.Counter(r['language'] for r in manifest)),
          'seed_counts': dict(collections.Counter(r['seed'] for r in manifest)),
          'condition_sha256': environment['condition_sha256'], 'sample_rate': 24000, 'channels': 1,
          'feedback_and_output_ids_match': True, 'all_output_hashes_match': True,
          'all_duration_windows_passed': True, 'quality_scores_computed': False, 'outputs': manifest}
(a.job / 'verified-outputs.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'outputs'}))
