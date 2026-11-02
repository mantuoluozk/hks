"""Audit optimized evidence. Historical accuracy remains attributed to the original release."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'scripts/summarize_optimized.py')],check=True)
v=json.loads((r/'results/optimized-summary.json').read_text())
assert v['measured_requests']==1536 and v['measured_input_tokens']==478720
assert v['embedding_api_pass'] and v['official_accuracy_pass'] is None
print(json.dumps({'optimized_evidence_audit':True,'independent_process_reproduction':None,'official_accuracy_pass':None}))
raise SystemExit(0 if a.selfcheck else 2)
