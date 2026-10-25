"""Keep local validation and the unavailable official GPU evaluation separate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--local-only',action='store_true',help='Validate completed local artifacts; does not claim official accuracy.')
args=parser.parse_args()
if args.local_only:
    script = 'finalize_scnet_results.py' if (root/'results/scnet/runtime/runtime-config.json').exists() else 'finalize_model.py'
    subprocess.run([sys.executable,str(root/'scripts'/script)],check=True,cwd=root)
    raise SystemExit(0)
(root/'results').mkdir(exist_ok=True)
result={'model_id':'BAAI/RoboBrain2.5-8B-NV','status':'official_reference_missing',
        'official_accuracy_pass':None,'official_scnet_validation':False,
        'required_consistency_percent':98,
        'execution_platform':'SCNet Notebook' if (root/'results/scnet/runtime/runtime-config.json').exists() else 'self-owned',
        'required':'Official image set (including depth), parser/comparison protocol, and per-case GPU action/coordinate references.',
        'local_summary_available':(root/'results/local_validation.json').exists()}
(root/'results/verify_result.json').write_text(json.dumps(result,indent=2))
(root/'verify_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(2)
