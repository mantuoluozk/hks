"""Recheck changed dependencies; retain the completed checkpoint hash evidence."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
reference=Path(os.environ['VERIFIED_WEIGHTS_RECORD'])
assert json.loads(reference.read_text())['all_runtime_files_match']
out=ROOT/'results/scnet'/('runtime-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True,exist_ok=False)
print(out,flush=True)
try:
    for stage,script in [('preflight','preflight.py'),('smoke512','smoke_pipeline.py')]:
        (out/'status.json').write_text(json.dumps({'stage':stage,'checkpoint_hash_evidence':str(reference.relative_to(ROOT)),'official_accuracy_pass':None}))
        with (out/(stage+'.log')).open('wb') as log:
            subprocess.run(['bash','run-scnet.sh','python','-u','scripts/'+script],cwd=ROOT,
                           env=dict(os.environ,RUN_RESULTS_DIR=str(out/stage)),stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'status.json').write_text(json.dumps({'stage':'compatibility_probe_complete','checkpoint_hash_evidence':str(reference.relative_to(ROOT)),'official_accuracy_pass':None}))
except BaseException as exc:
    (out/'status.json').write_text(json.dumps({'stage':'failed','error':repr(exc),'checkpoint_hash_evidence':str(reference.relative_to(ROOT)),'official_accuracy_pass':None}));raise
finally:
    with (out/'backup.log').open('wb') as log:
        subprocess.run(['python',str(ROOT.parent.parent/'platform/scnet-notebook/backup_shared.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
