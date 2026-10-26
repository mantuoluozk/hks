"""Keep local validation and the unavailable official GPU evaluation separate."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--local-only',action='store_true',help='Validate completed local artifacts; does not claim official accuracy.')
parser.add_argument('--selfcheck',action='store_true',help='Audit saved SCNet evidence without GPU inference or official acceptance claims.')
args=parser.parse_args()
if args.selfcheck:
    summary=json.loads((root/'results/scnet/validation-summary.json').read_text())
    assert summary['evidence_audit_passed'] and summary['official_accuracy_pass'] is None
    for item in summary['references']:
        path=root/item['path']
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['path']
    print(json.dumps({'saved_scnet_evidence':'PASS','references':len(summary['references']),
                      'official_accuracy_pass':None,'official_acceptance_passed':None}))
    raise SystemExit(0)
if args.local_only:
    subprocess.run([sys.executable,str(root/'scripts/finalize_model.py')],check=True,cwd=root)
    raise SystemExit(0)
(root/'results').mkdir(exist_ok=True)
result={'model_id':'Qwen/Qwen3-ASR-1.7B','status':'official_reference_missing',
        'official_accuracy_pass':None,'official_acceptance_passed':None,
        'required':'Official audio subset, text normalization/scoring protocol, and matching GPU WER/CER baseline.',
        'local_summary_available':(root/'results/local_validation.json').exists(),
        'scnet_summary_available':(root/'results/scnet/validation-summary.json').exists()}
(root/'results/verify_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
raise SystemExit(2)
