"""One owned final service: functionality, native context check, 608 requests, stop."""
import json,os,signal,socket,subprocess,time,shutil
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
selected=json.loads((ROOT/'results/selection.json').read_text())
assert selected['passed'] and selected['mode'] in ['eager','graph']
OUT=ROOT/'results'/time.strftime('formal-%Y%m%dT%H%M%SZ',time.gmtime());OUT.mkdir()
(ROOT/'results/formal-current.json').write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
(OUT/'source').mkdir()
for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.sh')):shutil.copyfile(p,OUT/'source'/p.name)
env=dict(os.environ,EXECUTION_MODE=selected['mode'],RUN_RESULTS_DIR=str(OUT),SMOKE_OUTPUT_DIR=str(OUT/'functional'))
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def state(stage,**extra):(OUT/'state.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
def command(cmd,name,timeout=3600):
    with (OUT/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def metrics(name):
    with op.open('http://127.0.0.1:'+os.environ['PORT']+'/metrics',timeout=10) as r:(OUT/(name+'.prom')).write_bytes(r.read())
proc=None
try:
    with (OUT/'launcher.log').open('w') as log:
        proc=subprocess.Popen(['bash','run.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state('loading',owner_pid=proc.pid)
        for _ in range(360):
            if proc.poll() is not None:raise RuntimeError(f'Final service launcher exit {proc.returncode}')
            if (OUT/'ready.json').exists():break
            time.sleep(5)
        else:raise TimeoutError('Readiness timeout')
        state('functional');command(['bash','smoke.sh'],'smoke',1800)
        state('context_probe')
        command(['python','scripts/context_probe.py','--output',str(OUT/'context-probe.json')],'context-probe',300)
        metrics('before-matrix')
        for length in [4096,16384]:
            for c in [1,8,64]:
                n=max(8,2*c)
                for repeat in [1,2]:
                    name=f'round{repeat}_{length}_1024_c{c}'
                    env['BENCH_INPUTS_DIR']=str(OUT/(name+'-raw'))
                    state('benchmarking',input_tokens=length,concurrency=c,repeat=repeat,requests=n)
                    cmd=['python','scripts/bench_native.py','--backend','vllm','--host','127.0.0.1','--port',os.environ['PORT'],
                         '--model',os.environ['MODEL_PATH'],'--tokenizer',os.environ['MODEL_PATH'],'--served-model-name',os.environ['SERVED_NAME'],
                         '--dataset-name','random-ids','--random-input-len',str(length),'--random-output-len','1024','--random-range-ratio','1',
                         '--max-concurrency',str(c),'--num-prompts',str(n),'--seed','1','--warmup-requests','1',
                         '--extra-request-body',json.dumps({'stream_options':{'include_usage':True},'top_p':1,'seed':42}),
                         '--output-details','--output-file',str(OUT/f'round{repeat}.jsonl')]
                    (OUT/(name+'-command.json')).write_text(json.dumps(cmd))
                    command(cmd,name,7200)
                    command(['python','scripts/check_cell.py','--file',str(OUT/f'round{repeat}.jsonl'),'--input',str(length),'--concurrency',str(c),'--raw',env['BENCH_INPUTS_DIR']],name+'-check',60)
                    metrics(name)
        metrics('after-matrix')
        command(['bash','stop.sh'],'stop',100);assert proc.wait(timeout=90)==0;proc=None
    state('completed',counted_requests=608,independent_service_reproduction=None,official_acceptance=None)
except BaseException as e:
    state('failed',error=repr(e),official_acceptance=None);raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=100)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
