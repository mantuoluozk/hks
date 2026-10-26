"""Check released evidence; official acceptance stays pending."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent
v=json.loads((root/'results/scnet/validation-summary.json').read_text())
assert v['evidence_audit_passed'] and v['functional_pass'] and v['embedding_api_pass']
for item in v['references']:
    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
for filename in ['upstream-source.json','qwen-vl-utils-source.json']:
    for item in json.loads((root/'sources'/filename).read_text())['files']:
        assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert v['official_accuracy_pass'] is None
print(json.dumps({'saved_evidence_integrity':True,'performance_reproduction_within_20_percent':v['performance']['within_20_percent'],'official_accuracy_pass':None}))
raise SystemExit(0 if args.selfcheck else 2)
