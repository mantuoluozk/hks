"""One controlled C4 warmup experiment, sequential fresh eager/graph services."""
import concurrent.futures as cf
import hashlib,json,os,re,shutil,socket,statistics,subprocess,threading,time
from pathlib import Path
from asr_api import audio,infer
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT=ROOT/'results'/time.strftime('c4-warmup-%Y%m%dT%H%M%SZ',time.gmtime())
OUT.mkdir()
(ROOT/'results/c4-warmup-current.json').write_text(json.dumps({'path':str(OUT),'controller_pid':os.getpid()}))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def state(stage,**kw):write(OUT/'state.json',{'stage':stage,**kw})
def run(cmd,log,env,timeout):
    with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
def compiled(lines):return [l for l in lines.splitlines() if re.search(r'took .* to compile|compilation can stall',l,re.I)]
items=[]
for lang in ['en','zh']:
    p=ROOT/f'data/examples/asr_{lang}.wav'
    if not p.exists():p=Path(os.environ['OLD_ASR_ROOT'])/f'data/examples/asr_{lang}.wav'
    items.append({**audio(p),'language_input':lang})
write(OUT/'protocol.json',{'concurrency':4,'requests_per_batch':4,'warmup_batches':3,'measured_batches':3,'counted_requests_per_mode':12,'warmup_requests_per_mode':12,'batch_order':['en','zh','en','zh'],'start_barrier_threads':4,'fresh_services':['eager','graph'],'disk_compile_cache':'Existing shared model-local disk compiler cache, retained; this measures warmed execution, not cold-start causality','only_execution_mode_changed':True,'timing':'Pool lifecycle, barrier and HTTP encoding/decoding included; audio/base64 before timing, evidence writes after batch','acceptance':'Both last two warmup batches and all three measured batches have no serving-time compilation warnings; every output matches the original fixture text; no errors/truncations'})
source=OUT/'source';source.mkdir()
for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.sh')):
    shutil.copyfile(p,source/p.name)
write(OUT/'source-hashes.json',[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source.iterdir())])
reference={lang:set() for lang in ['en','zh']}
for line in (ROOT/'results/eager-template/short/requests.jsonl').read_text().splitlines():
    r=json.loads(line);reference[r['language_input']].add(r['text'])
assert all(len(s)==1 for s in reference.values())
cells={}
try:
    for mode in ['eager','graph']:
        job=OUT/mode;job.mkdir()
        env=dict(os.environ,EXECUTION_MODE=mode,RUN_RESULTS_DIR=str(job))
        proc=None;all_rows=[];batches=[]
        try:
            with (job/'launcher.log').open('w') as log:
                proc=subprocess.Popen(['bash','run.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state('loading',mode=mode,owner_pid=proc.pid)
                for _ in range(360):
                    if proc.poll() is not None:raise RuntimeError(f'{mode} launcher exited {proc.returncode}')
                    if (job/'ready.json').exists():break
                    time.sleep(5)
                else:raise TimeoutError('Service readiness')
                state('batches',mode=mode)
                for phase in ['warmup','measured']:
                    for repeat in [1,2,3]:
                        before=(job/'server.log').stat().st_size
                        barrier=threading.Barrier(4)
                        def request(item):
                            barrier.wait(timeout=30)
                            return infer(item,persist=(phase=='warmup' and repeat==1))
                        started=time.perf_counter();start_unix=time.time()
                        with cf.ThreadPoolExecutor(4) as pool:rows=list(pool.map(request,[items[i%2] for i in range(4)]))
                        wall=time.perf_counter()-started
                        for i,r in enumerate(rows):
                            r.update(phase=phase,batch=repeat,request_index=i)
                            all_rows.append(r)
                            if phase=='warmup' and repeat==1:items[i%2]['_request_ref']=r['request']
                        (job/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows),encoding='utf-8')
                        assert all(r['success'] and not r['error'] and r['finish_reason']!='length' and r['text'] in reference[r['language_input']] and r['usage']['completion_tokens']>0 for r in rows),'Failed/changed/truncated response'
                        time.sleep(.25)
                        segment=(job/'server.log').read_bytes()[before:];name=f'{phase}-{repeat}.server.log';(job/name).write_bytes(segment)
                        cell={'phase':phase,'batch':repeat,'start_unix':start_unix,'concurrency':4,'requests':4,'wall_seconds':wall,'audio_seconds':sum(r['audio_seconds'] for r in rows),'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),'compilation_warnings':compiled(segment.decode(errors='replace')),'server_log_segment':name}
                        cell['audio_seconds_per_wall_second']=cell['audio_seconds']/wall
                        batches.append(cell);write(job/'batches.json',batches)
                        print(mode,phase,repeat,cell['audio_seconds_per_wall_second'],'compile_warnings',len(cell['compilation_warnings']),flush=True)
                stable=all(not r['compilation_warnings'] for r in batches if r['phase']=='measured' or r['batch'] in [2,3])
                assert stable,'Compilation still occurred after warmup; do not claim warmed comparison'
                measured=[r for r in batches if r['phase']=='measured'];rows=[r for r in all_rows if r['phase']=='measured'];wall=sum(r['wall_seconds'] for r in measured)
                result={'passed':True,'mode':mode,'concurrency':4,'counted_requests':12,'warmup_requests':12,'wall_seconds':wall,'audio_seconds':sum(r['audio_seconds'] for r in rows),'mean_latency_seconds':statistics.mean(r['elapsed_seconds'] for r in rows),'mean_request_rtf':statistics.mean(r['rtf'] for r in rows),'compilation_free_last_two_warmups_and_measurement':stable,'text_matches_original_fixture':True,'batch_throughputs':[r['audio_seconds_per_wall_second'] for r in measured]}
                result['audio_seconds_per_wall_second']=result['audio_seconds']/wall;cells[mode]=result;write(job/'summary.json',result)
        finally:
            if proc is not None and proc.poll() is None:
                run(['bash','stop.sh'],job/'stop.log',env,120)
                assert proc.wait(timeout=100)==0
    write(OUT/'summary.json',{'passed':True,'cells':cells,'graph_over_eager_percent':100*(cells['graph']['audio_seconds_per_wall_second']/cells['eager']['audio_seconds_per_wall_second']-1),'scope':'C4 warmed short comparison only; C1, 128 formal performance, 2620 accuracy and official300 are unchanged','official_acceptance':None})
    state('completed')
except BaseException as e:
    state('failed',error=repr(e));raise
