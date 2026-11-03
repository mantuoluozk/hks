"""Warm real audio; natural EOS. Timed batch has no request/evidence file I/O."""
import argparse,concurrent.futures as cf,json,os,statistics,time
from pathlib import Path
import numpy as np
from asr_api import audio,infer
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--short',action='store_true');a=p.parse_args();a.output.mkdir()
old=Path(os.environ['OLD_ASR_ROOT']);items=[]
for lang in ['en','zh']:
    path=ROOT/f'data/examples/asr_{lang}.wav'
    if not path.exists():path=old/f'data/examples/asr_{lang}.wav'
    items.append({**audio(path),'language_input':lang})
cells=[];details=[];warmups=[]
for c in ([1,4] if a.short else [1,4,8,16]):
    count=4 if a.short else max(8,2*c)
    for repeat in ([1] if a.short else [1,2]):
        for item in items:
            r=infer(item);warmups.append({'concurrency':c,'round':repeat,**r})
            (a.output/'warmups.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in warmups))
            assert r['success'] and r['text'] and r['finish_reason']!='length',r.get('error')
            item['_request_ref']=r['request']
        started=time.perf_counter()
        with cf.ThreadPoolExecutor(c) as pool:rows=list(pool.map(lambda item:infer(item,persist=False),[items[i%2] for i in range(count)]))
        wall=time.perf_counter()-started
        for i,r in enumerate(rows):details.append({'round':repeat,'concurrency':c,'request':i,**r})
        (a.output/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
        assert all(r['success'] and r['text'] and r.get('finish_reason')!='length' and r.get('usage',{}).get('completion_tokens',0)>0 for r in rows),'Audio request failure/empty/truncation/usage missing'
        total=sum(r['audio_seconds'] for r in rows);lat=[r['elapsed_seconds'] for r in rows]
        cell={'round':repeat,'concurrency':c,'requests':count,'wall_seconds':wall,'audio_seconds':total,'audio_seconds_per_wall_second':total/wall,
              'mean_request_rtf':statistics.mean(r['rtf'] for r in rows),'mean_latency_seconds':statistics.mean(lat),
              'p95_latency_seconds':float(np.percentile(lat,95)),'p99_latency_seconds':float(np.percentile(lat,99)),'max_latency_seconds':max(lat),
              'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),'errors':sum(not r['success'] for r in rows)}
        cells.append(cell);print(json.dumps(cell),flush=True)
        (a.output/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
        (a.output/'warmups.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in warmups))
        (a.output/f'round{repeat}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cells if r['round']==repeat))
summary={'passed':True,'short':a.short,'counted_requests':len(details),'warmup_requests':len(warmups),'round_reported':1 if a.short else 2,'cells':cells,
         'text_consistent_by_fixture':{lang:len({r['text'] for r in details if r['language_input']==lang})==1 for lang in ['en','zh']},
         'decode':'BF16 greedy max1024 natural EOS automatic language','timing':'Client end-to-end wall includes pool lifecycle and HTTP encoding/decoding; audio bytes/base64 prepared before timing; raw file writes after batch',
         'official_performance_protocol':False,'independent_service_full_reproduction':None}
assert len(details)==(8 if a.short else 128) and len(warmups)==(4 if a.short else 16)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
