"""Run one isolated ForcedAligner validation and preserve failure evidence."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

p=Path(__file__).resolve().parents[1]
r=p/'results/scnet'/('validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
r.mkdir(parents=True,exist_ok=False)
print(str(r),flush=True)


def run(stage,args,dest):
    (r/'status.json').write_text(json.dumps({'stage':stage,'official_accuracy_pass':None}))
    with (r/(stage+'.log')).open('wb') as f:
        subprocess.run(['bash','run-scnet.sh']+args,cwd=p,
                       env=dict(os.environ,RUN_RESULTS_DIR=str(r/dest)),
                       stdout=f,stderr=subprocess.STDOUT,check=True)


try:
    run('weights',['python','scripts/check_weights.py'],'runtime')
    run('preflight',['python','scripts/preflight.py'],'preflight')
    run('offline_alignment',['python','scripts/smoke_offline.py'],'smoke')
    stage='functional_and_performance_complete' if os.environ.get('BATCH_BENCH')=='1' else 'initial_functional_complete'
    (r/'status.json').write_text(json.dumps({'stage':stage,'official_accuracy_pass':None}))
except BaseException as e:
    (r/'status.json').write_text(json.dumps({'stage':'failed','error':type(e).__name__+': '+str(e),'official_accuracy_pass':None}))
    raise
finally:
    with (r/'backup.log').open('wb') as f:
        subprocess.run(['python',str(p.parent.parent/'platform/scnet-notebook/backup_shared.py')],
                       stdout=f,stderr=subprocess.STDOUT,check=True)
