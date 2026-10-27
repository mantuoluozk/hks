"""Run checkpoint integrity, runtime imports and first full pipeline serially."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/scnet'/('validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True,exist_ok=False)
print(out,flush=True)
try:
    for stage,script in [('weights','check_weights.py'),('preflight','preflight.py'),('smoke512','smoke_pipeline.py')]:
        (out/'status.json').write_text(json.dumps({'stage':stage,'official_accuracy_pass':None}))
        with (out/(stage+'.log')).open('wb') as log:
            subprocess.run(['bash','run-scnet.sh','python','-u','scripts/'+script],cwd=ROOT,
                           env=dict(os.environ,RUN_RESULTS_DIR=str(out/stage)),stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'status.json').write_text(json.dumps({'stage':'compatibility_probe_complete','official_accuracy_pass':None}))
except BaseException as exc:
    (out/'status.json').write_text(json.dumps({'stage':'failed','error':repr(exc),'official_accuracy_pass':None}));raise
finally:
    with (out/'backup.log').open('wb') as log:
        subprocess.run(['python',str(ROOT.parent.parent/'platform/scnet-notebook/backup_shared.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
