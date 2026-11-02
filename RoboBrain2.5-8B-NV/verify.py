"""Read-only audit of saved SGLang evidence; no model or report generation."""
import argparse
import json
import math
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--local-only',action='store_true',help='Compatibility alias for the default saved-evidence audit.')
parser.add_argument('--selfcheck',action='store_true',help='Compatibility alias for the default saved-evidence audit.')
args=parser.parse_args()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

summary = read(root/'results/scnet/validation-summary.json')
smoke = read(root/summary['smoke']['path']/'summary.json')
assert smoke['functional_pass'] and len(smoke['cases']) == 8
assert all(row['functional_pass'] for row in smoke['cases'])
base = root/summary['performance']['path']
audit = read(base/'audit.json')
assert audit['complete'] and len(audit['workloads']) == 12
count = 0
for repeat in (1, 2):
    rows = [json.loads(line) for line in (base/f'round{repeat}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 6
    assert {(row['random_input_len'], row['max_concurrency']) for row in rows} == {(n, c) for n in (4096, 16384) for c in (1, 8, 64)}
    for row in rows:
        requests = max(8, 2*row['max_concurrency'])
        assert row['completed'] == requests and row['total_output_tokens'] == requests*1024
        assert row['duration'] > 0
        assert math.isclose(row['output_throughput'], row['total_output_tokens']/row['duration'], rel_tol=1e-6)
        saved = next(item for item in audit['workloads'] if (item['round'], item['input'], item['concurrency']) == (repeat, row['random_input_len'], row['max_concurrency']))
        assert saved['completed'] == requests and saved['generated_tokens'] == row['total_output_tokens']
        count += requests
assert count == summary['performance']['requests'] == 608
assert summary['official_accuracy_pass'] is None and summary['official_acceptance_passed'] is None
print(json.dumps({'saved_evidence_verified': True, 'functional_checks': 8, 'matrix_requests': count,
                  'official_accuracy_pass': None, 'official_acceptance_passed': None}))
