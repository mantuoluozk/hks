"""Verify saved evidence; never label format checks as official AAS accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');args=p.parse_args()
if args.selfcheck:
    s=json.loads((ROOT/'results/scnet/validation-summary.json').read_text())
    assert s['evidence_audit_passed'] and s['official_accuracy_pass'] is None
    for item in s['references']:
        f=ROOT/item['path'];assert f.resolve().is_relative_to(ROOT)
        assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],item['path']
    print(json.dumps({'saved_evidence':'PASS','references':len(s['references']),'official_accuracy_pass':None}))
    raise SystemExit(0)
print(json.dumps({'status':'awaiting_official_alignment_evaluation','official_accuracy_pass':None,
                  'required':'Matching annotated word/character boundaries, scoring protocol and official GPU AAS baseline.'}))
raise SystemExit(2)
