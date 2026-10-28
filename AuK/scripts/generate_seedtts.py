"""Generate a fixed Seed-TTS-Eval English split without filtering examples."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--max-samples',type=int)
p.add_argument('--seed',type=int,default=1234)
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if (out/'generated.jsonl').exists():raise RuntimeError('Use a new output directory; do not overwrite measurements')
data=ROOT/'data/seed-tts-eval-arrow/data/en-00000-of-00001.parquet'
digest=hashlib.sha256(data.read_bytes()).hexdigest()
assert digest=='5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca'
rows=pq.read_table(data).to_pylist()
assert len(rows)==1088 and len({r['sample_id'] for r in rows})==1088
selected=rows[:a.max_samples] if a.max_samples is not None else rows
manifest={'dataset':'zhaochenyang20/seed-tts-eval-arrow',
          'revision':'27f4c1adee83b5b29b7c4b375f6b976324bda308','parquet_sha256':digest,
          'split':'en','total_rows':len(rows),'selected_rows':len(selected),
          'full_dataset':len(selected)==len(rows),'seed':a.seed,'nfe':32,'cfg_strength':2.0,
          'conditioning_dtype':'bfloat16','dit_dtype':'float32','attention_backend':'math',
          'ids':[r['sample_id'] for r in selected]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
results=[]
with (out/'generated.jsonl').open('w') as log:
    for index,row in enumerate(selected):
        result={'index':index,'id':row['sample_id'],'reference_text':row['ref_text'],
                'target_text':row['target_text'],'seed':a.seed,'passed':False}
        try:
            assert row['ref_text'].strip() and row['target_text'].strip()
            raw=row['ref_audio']['bytes']
            info=sf.info(io.BytesIO(raw))
            estimate=info.duration*len(row['target_text'].encode('utf-8'))/len(row['ref_text'].encode('utf-8'))
            duration=min(estimate,30.0)
            assert duration>0
            result.update(reference_audio_sha256=hashlib.sha256(raw).hexdigest(),
                          reference_seconds=info.duration,estimated_seconds=estimate,
                          requested_seconds=duration,duration_capped=estimate>30)
            body={'input':row['target_text'],'ref_audio':'data:audio/wav;base64,'+base64.b64encode(raw).decode(),
                  'ref_text':row['ref_text'],'seed':a.seed,'response_format':'wav',
                  'stage_params':{'auk_engine':{'gen_seconds':duration}}}
            started=time.perf_counter()
            response=requests.post('http://127.0.0.1:30005/v1/audio/speech',json=body,timeout=300)
            result.update(status=response.status_code,seconds=time.perf_counter()-started)
            response.raise_for_status()
            audio,rate=sf.read(io.BytesIO(response.content),dtype='float32')
            assert rate==24000 and audio.ndim==1 and np.isfinite(audio).all()
            assert float(np.std(audio))>1e-6
            expected=math.ceil(duration*50)*480
            assert abs(len(audio)-expected)<=480
            path=out/f'{index:04d}.wav';path.write_bytes(response.content)
            result.update(wav_file=path.name,audio_seconds=len(audio)/rate,samples=len(audio),sample_rate=rate,
                          response_sha256=hashlib.sha256(response.content).hexdigest(),passed=True)
        except Exception as error:
            result['error']=type(error).__name__+': '+str(error)
        log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush();results.append(result)
        summary={'selected_rows':len(selected),'completed':len(results),'errors':sum(not r['passed'] for r in results),
                 'full_dataset':manifest['full_dataset'],'generation_complete':len(results)==len(selected),
                 'official_accuracy_tested':False}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if index%25==0 or not result['passed'] or index+1==len(selected):print(json.dumps(summary),flush=True)
raise SystemExit(0 if all(r['passed'] for r in results) else 2)
