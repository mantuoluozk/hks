"""All 2620 requests; errors/truncation remain in denominator and fail technically."""
import argparse,concurrent.futures as cf,json,time
from pathlib import Path
from asr_api import audio,infer
from score_asr import score,summary
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
items=[json.loads(s) for s in (ROOT/'data/test-clean-items.jsonl').read_text().splitlines()];assert len(items)==2620
(a.output/'protocol.json').write_text(json.dumps({'workers':4,'temperature':0,'max_tokens':1024,'language':None,'dtype':'bfloat16','natural_eos':True,'source_sha256':'7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0','reference_text_in_request':False,'normalizer_commit':'86098128c0b4f24f0e2aa2994de830614b474227'},indent=2)+'\n')
def run(item):
    r=infer(audio(item['audio_path']));return {'id':item['id'],**r,**score(item['reference'],r['text'])}
rows=[];started=time.perf_counter()
with cf.ThreadPoolExecutor(4) as pool,(a.output/'predictions.jsonl').open('w',encoding='utf-8') as f:
    futures=[pool.submit(run,item) for item in items]
    for future in cf.as_completed(futures):
        r=future.result();rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
        if len(rows)%100==0:print('Evaluated',len(rows),'of 2620; request errors',sum(not r['success'] for r in rows),flush=True)
report=summary(rows);report['wall_seconds']=time.perf_counter()-started
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
raise SystemExit(0 if report['complete'] else 1)
