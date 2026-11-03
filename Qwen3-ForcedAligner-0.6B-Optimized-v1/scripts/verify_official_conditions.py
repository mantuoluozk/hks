"""Validate the official 30-row forced-alignment output package without computing AAS."""
import argparse
import hashlib
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--job', type=Path, required=True)
p.add_argument('--conditions', type=Path, required=True)
a = p.parse_args()
lock=json.loads((Path(__file__).resolve().parents[1]/'sources/official-t03t04-20261030.json').read_text(encoding='utf8'))
assert hashlib.sha256(a.conditions.read_bytes()).hexdigest()==lock['condition_sha256']
conditions = [json.loads(line) for line in a.conditions.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
expected = {row['id']: row for row in conditions}
assert len(expected) == len(conditions) == 30
summary = json.loads((a.job / 'summary.json').read_text(encoding='utf-8'))
assert summary['expected'] == summary['completed'] == summary['successful'] == 30
assert summary['failed'] == 0 and summary['boundary_accuracy_computed'] is False
with (a.job / '反馈表.csv').open(encoding='utf-8-sig', newline='') as stream:
    feedback = list(csv.DictReader(stream))
assert len(feedback) == len({row['condition_id'] for row in feedback}) == 30
assert {row['condition_id'] for row in feedback} == set(expected)
assert all(row['model'] == 'forced_aligner' and row['status'] == '成功' and not row['error'] and row['parameter_changes'] == '无' for row in feedback)
assert {row['output_path'] for row in feedback} == {'outputs/' + key + '.json' for key in expected}
for condition_id, condition in expected.items():
    record = json.loads((a.job / 'outputs' / (condition_id + '.json')).read_text(encoding='utf-8'))
    assert record['id'] == condition_id and record['language'] == 'English'
    assert record['evaluation_status'] == 'BOUNDARIES_PENDING' and record['timestamp_unit'] == 'seconds' and record['seed'] == 42
    assert record['audio_sha256']==condition['assets']['audio']['sha256'] and record['condition_sha256']==lock['condition_sha256']
    items = record['items']
    assert items and ' '.join(item['text'] for item in items) == condition['text']
    times = []
    for item in items:
        start, end = item['start_time'], item['end_time']
        assert isinstance(item['text'], str) and item['text']
        assert math.isfinite(start) and math.isfinite(end) and 0 <= start <= end
        times.extend((start, end))
    assert all(left <= right for left, right in zip(times, times[1:]))
    assert max(times) <= record['audio_seconds'] + 0.16
    assert record['format_monotonicity_bounds_passed'] is True
result = {'expected': 30, 'completed': 30, 'successful': 30, 'failed': 0,
          'outputs_verified': 30, 'condition_ids_match': True, 'feedback_rows_match': True,
          'transcript_words_match_inputs': True, 'timestamp_unit': 'seconds',
          'timestamps_monotonic_and_in_audio_bounds': True,
          'boundary_accuracy_computed': False, 'evaluation_status': 'BOUNDARIES_PENDING'}
(a.job / 'verified-summary.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps(result, ensure_ascii=False))
