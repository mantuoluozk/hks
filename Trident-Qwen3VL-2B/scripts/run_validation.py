"""Validate publisher reference and native embeddings sequentially on one BW card."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/scnet'/('validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True,exist_ok=False)
print(str(out),flush=True)


def run(stage,script,env_extra=None):
    (out/'status.json').write_text(json.dumps({'stage':stage,'official_accuracy_pass':None}))
    env=dict(os.environ,RUN_RESULTS_DIR=str(out/stage),**(env_extra or {}))
    with (out/(stage+'.log')).open('wb') as f:
        subprocess.run(['bash','run-scnet.sh','python','scripts/'+script],cwd=ROOT,env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)


try:
    run('weights','check_weights.py')
    run('preflight','preflight.py')
    run('publisher_reference','reference_smoke.py')
    run('native_embedding','native_smoke.py',{'REFERENCE_RESULTS':str(out/'publisher_reference/summary.json'),'RUN_BENCHMARK':'1'})
    (out/'status.json').write_text(json.dumps({'stage':'initial_native_reference_comparison_complete','official_accuracy_pass':None}))
except BaseException as e:
    (out/'status.json').write_text(json.dumps({'stage':'failed','error':type(e).__name__+': '+str(e),'official_accuracy_pass':None}))
    raise
finally:
    with (out/'backup.log').open('wb') as f:
        subprocess.run(['python',str(ROOT.parent.parent/'platform/scnet-notebook/backup_shared.py')],
                       stdout=f,stderr=subprocess.STDOUT,check=True)
