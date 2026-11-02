"""Recompute published evidence and the documented 2-point submission threshold."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'results/accuracy/ocrbench-16x-review-20261030T060837Z'


def within_tolerance(got, reference, tolerance):
    return Decimal(str(reference)) - Decimal(str(got)) <= Decimal(str(tolerance))


p = argparse.ArgumentParser(__doc__)
p.add_argument('--selfcheck', action='store_true')
a = p.parse_args()
if a.selfcheck:
    assert within_tolerance(88.4, 90.4, 2)
    assert not within_tolerance(88.3, 90.4, 2)
    assert within_tolerance(91, 90.4, 2)
    print('threshold selfcheck passed; checking saved evaluation evidence')
summary = json.loads((JOB / 'ocrbench/summary.json').read_text())
audit = json.loads((JOB / 'independent-audit.json').read_text())
assert summary['completed'] == summary['expected'] == audit['rows'] == 1000
assert summary['correct'] == audit['correct'] == 888
assert summary['responses_sha256'] == audit['raw_sha256']
manifest = ROOT / 'sources/release-files.json'
if manifest.exists():
    for row in json.loads(manifest.read_text())['files']:
        path = ROOT / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
perf = json.loads((ROOT / 'results/validation-20261030T025528Z/summary.json').read_text())
assert perf['performance_reproduced'] is True
got = Decimal(summary['correct']) * 100 / Decimal(summary['expected'])
result = {'model': 'MiniCPM-V-4.7-35B-A3B', 'dataset': 'OCRBench', 'rows': 1000,
          'got': float(got), 'reference': 90.4, 'worse_points': float(Decimal('90.4') - got),
          'tolerance_points': 2.0, 'precision_numeric_pass': within_tolerance(got, 90.4, 2),
          'performance_reproduced': True, 'functional_checks': '8/8 in each of two independent services',
          'reference_kind': 'publisher 16x instruct public score', 'protocol': 'declared local protocol; full publisher harness unavailable',
          'clean_environment_scope': 'same Notebook, independent service processes; no cross-Notebook claim',
          'official_acceptance': None, 'http_errors': audit['http_errors'], 'truncated_responses': audit['length']}
(ROOT / 'verify_result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result))
raise SystemExit(0 if result['precision_numeric_pass'] else 2)
