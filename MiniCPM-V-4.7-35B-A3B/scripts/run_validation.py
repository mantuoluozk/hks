"""Run two separately owned services. Use only after weights are complete."""
import hashlib, json, os, signal, socket, statistics, subprocess, time, urllib.request, shutil, fcntl
from datetime import datetime, timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lock=(root/'results/validation.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
port=int(os.environ.get('PORT','30147'))
job=root/'results'/('validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
job.mkdir(parents=True)
(root/'results/validation-current.json').write_text(json.dumps({'job':str(job),'controller_pid':os.getpid()},indent=2)+'\n')
def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Interrupted by signal {signum}')
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
for folder in ('scripts','python','sources'):
    for src in (root/folder).rglob('*'):
        if src.is_file() and '__pycache__' not in src.parts:
            dst=job/'source'/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for src in root.glob('*.sh'):
    shutil.copy2(src,job/'source'/src.name)


def save(name,value):
    (job/name).write_text(json.dumps(value,indent=2)+'\n')

def stop(proc):
    # Popen created this session; terminate only that process group and descendants.
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=30)
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass

save('state.json',{'stage':'starting','official_accuracy_tested':False})
try:
    for round_id in (1,2):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Port already occupied; refusing to reuse an existing service')
        dest=job/f'round{round_id}';dest.mkdir()
        with (dest/'server.log').open('w') as log:
            proc=subprocess.Popen(['bash','run.sh'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (dest/'owned-server.pid').write_text(str(proc.pid)+'\n')
            try:
                deadline=time.monotonic()+2400
                while True:
                    if proc.poll() is not None:raise RuntimeError(f'Owned server exited: {proc.returncode}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as r:assert r.status==200
                        break
                    except Exception:
                        if time.monotonic()>deadline:raise TimeoutError('Owned server startup timed out')
                        time.sleep(2)
                save('state.json',{'stage':'running','round':round_id,'owned_pid':proc.pid})
                for mode in ('smoke','bench'):
                    subprocess.run(['bash',f'{mode}.sh',str(dest)],cwd=root,check=True)
            finally:stop(proc)
    summaries=[]
    for size in (448,896):
        for concurrency in (1,2,4):
            means=[]
            for r in (1,2):
                rows=[json.loads(s) for s in (job/f'round{r}'/'requests.jsonl').read_text().splitlines()]
                values=[x['seconds'] for x in rows if not x['warmup'] and x['size']==size and x['concurrency']==concurrency]
                assert len(values)==3*concurrency
                means.append(statistics.mean(values))
            deviation=abs(means[1]-means[0])/means[0]
            summaries.append({'size':size,'concurrency':concurrency,'means':means,'deviation':deviation,'within_20_percent':deviation<=.2})
    save('summary.json',{'functional_rounds':2,'performance':summaries,'performance_reproduced':all(x['within_20_percent'] for x in summaries),'official_accuracy_tested':False})
    save('state.json',{'stage':'completed'})
except BaseException as e:
    save('state.json',{'stage':'failed','error':repr(e)})
    raise
finally:
    with (job/'backup.log').open('w') as f:subprocess.run(['python',str(root.parents[1]/'platform/scnet-notebook/backup_shared.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
