"""Serialize reference retrieval and two independent native performance processes."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/scnet'/('release-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True,exist_ok=False)
print(out,flush=True)
try:
    for name,command,extra in [
        ('publisher_chartqa',['evaluate_chartqa.py','--backend','publisher'],{}),
        ('native_primary',['run_native_suite.py'],{}),
        ('native_reproduction',['run_native_suite.py'],{'SKIP_CHARTQA':'1'}),
    ]:
        (out/'status.json').write_text(json.dumps({'stage':name,'official_accuracy_pass':None}))
        env=dict(os.environ,RUN_RESULTS_DIR=str(out/name),**extra)
        with (out/(name+'.log')).open('wb') as log:
            subprocess.run(['bash','run-scnet.sh','python','-u','scripts/'+command[0],*command[1:]],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'status.json').write_text(json.dumps({'stage':'completed','official_accuracy_pass':None}))
except BaseException as exc:
    (out/'status.json').write_text(json.dumps({'stage':'failed','error':repr(exc),'official_accuracy_pass':None}))
    raise
finally:
    with (out/'backup.log').open('wb') as log:
        subprocess.run(['python',str(ROOT.parent.parent/'platform/scnet-notebook/backup_shared.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
