"""Check saved platform evidence; official accuracy awaits organizer code."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if '--selfcheck' in sys.argv or len(sys.argv) == 1:
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_scnet.py'),
                    str(ROOT/'results/scnet/validation-20261026T021625Z')],check=True)
    raise SystemExit(0)
result=json.loads((ROOT/'verify_result.json').read_text())
assert result['official_accuracy_pass'] is None
print(json.dumps(result,ensure_ascii=False,indent=2))
print('Official accuracy awaits organizer evaluation code; no PASS is asserted.')
raise SystemExit(2)
