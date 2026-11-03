"""Chat, transcription and output SSE are separate from real-time input."""
import concurrent.futures as cf,json,os
from pathlib import Path
from asr_api import audio,infer
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ['SMOKE_OUTPUT_DIR']);OUT.mkdir(parents=True)
old=Path(os.environ['OLD_ASR_ROOT']);items={lang:audio(ROOT/f'data/examples/asr_{lang}.wav' if (ROOT/f'data/examples/asr_{lang}.wav').exists() else old/f'data/examples/asr_{lang}.wav') for lang in ['en','zh']};rows=[];plain={}
def save(name,lang,r):
    valid=r['success'] and bool(r['text']) and r.get('finish_reason')!='length'
    valid=valid and (any('\u4e00'<=x<='\u9fff' for x in r['text']) if lang=='zh' else any(x.isascii() and x.isalpha() for x in r['text']))
    r.update(id=name,language_input=lang,passed=valid);rows.append(r);(OUT/(name+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(name,valid,flush=True);return r
for lang in ['en','zh']:
    plain[lang]=save(lang+'-chat',lang,infer(items[lang]))
    save(lang+'-transcription',lang,infer(items[lang],'transcription'))
    save(lang+'-sse',lang,infer(items[lang],'sse'))
    save(lang+'-repeat',lang,infer(items[lang]))
with cf.ThreadPoolExecutor(4) as pool:
    futures=[(lang,pool.submit(infer,items[lang])) for lang in ['en','zh','en','zh']]
    for i,(lang,future) in enumerate(futures):save('concurrent-'+str(i),lang,future.result())
summary={'passed':all(r['passed'] for r in rows),'total':len(rows),'successful':sum(r['passed'] for r in rows),
         'repeat_exact':{lang:next(r['text'] for r in rows if r['id']==lang+'-repeat')==plain[lang]['text'] for lang in items},
         'sse_chat_exact':{lang:next(r['text'] for r in rows if r['id']==lang+'-sse')==plain[lang]['text'] for lang in items},
         'native_transcription_sampling':'Native adapter changes temperature=0 to 0.01 and caps max_new_tokens=256; not the primary accuracy/performance path',
         'stream_scope':'Complete offline audio input; SSE output chunks only. No real-time input verification','official_acceptance':None}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
raise SystemExit(0 if summary['passed'] else 1)
