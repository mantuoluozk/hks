"""Finish the owned eager service, then one graph candidate; no long benchmark."""
import json,os,subprocess,time,shutil,math,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'results'/time.strftime('screen-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir()
(ROOT/'results/screen-current.json').write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
def state(stage,**extra):(OUT/'state.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
def command(cmd,log,env,timeout):
    with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
results={};paths={}
try:
    for mode in ['eager','graph']:
        job=ROOT/'results/eager-template' if mode=='eager' else OUT/'graph';job.mkdir(exist_ok=True)
        paths[mode]=str(job.relative_to(ROOT));env=dict(os.environ,EXECUTION_MODE=mode,RUN_RESULTS_DIR=str(job),SMOKE_OUTPUT_DIR=str(job/'functional'))
        source=job/'source';source.mkdir()
        for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.sh')):shutil.copyfile(p,source/p.name)
        proc=None
        try:
            if mode=='eager':
                current=json.loads((ROOT/'results/server-current.json').read_text());assert Path(current['path'])==job and (job/'ready.json').exists()
            else:
                log=(job/'launcher.log').open('w');proc=subprocess.Popen(['bash','run.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state('loading',candidate=mode,owner_pid=proc.pid)
                for _ in range(360):
                    if proc.poll() is not None:raise RuntimeError(f'Candidate launcher exited {proc.returncode}')
                    if (job/'ready.json').exists():break
                    time.sleep(5)
                else:raise TimeoutError('Graph readiness')
            state('functional',candidate=mode);command(['bash','smoke.sh'],job/'smoke.log',env,1800)
            state('short',candidate=mode);command(['python','scripts/bench_audio.py','--short','--output',str(job/'short')],job/'short.log',env,1800)
            results[mode]=json.loads((job/'short/summary.json').read_text());assert results[mode]['passed']
        finally:
            command(['bash','stop.sh'],job/'stop.log',env,120)
            if proc is not None:assert proc.wait(timeout=100)==0;log.close()
    def texts(mode):
        rows=[json.loads(s) for s in (ROOT/paths[mode]/'short/requests.jsonl').read_text().splitlines()]
        return {lang:sorted({r['text'] for r in rows if r['language_input']==lang}) for lang in ['en','zh']}
    same=texts('eager')==texts('graph');ratios=[g['audio_seconds_per_wall_second']/e['audio_seconds_per_wall_second'] for e,g in zip(results['eager']['cells'],results['graph']['cells'])]
    selected='graph' if same and all(r>=1 for r in ratios) else 'eager'
    decision={'passed':True,'mode':selected,'screen':OUT.name,'candidate_paths':paths,'same_short_text_sets':same,'graph_over_eager_ratios':ratios,'short_protocol':'c1/c4, four counted requests per cell, two audio warmups per cell, one pass per candidate'}
    (ROOT/'results/selection.json').write_text(json.dumps(decision,indent=2)+'\n');state('completed',**decision)
except BaseException as e:state('failed',error=repr(e));raise
