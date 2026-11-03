"""Verify existing audio against fixed sources; no download or audio rewriting."""
import hashlib,json,os
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];OLD=Path(os.environ['OLD_ASR_ROOT'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def existing(own,old,variable):return Path(os.environ.get(variable,str(own if own.exists() else old)))
parquet=existing(ROOT/'data/librispeech-source/test-clean.parquet',OLD/'data/librispeech-source/test-clean.parquet','LIBRI_PARQUET')
assert parquet.stat().st_size==350452636 and sha(parquet)=='7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0'
folder=existing(ROOT/'data/LibriSpeech/test-clean',OLD/'data/LibriSpeech/test-clean','LIBRI_TEST_CLEAN')
references={}
for p in folder.rglob('*.trans.txt'):
    for line in p.read_text().splitlines():
        uid,text=line.split(' ',1);assert uid not in references;references[uid]=text
assert len(references)==2620
proof=[];items=[]
for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64,columns=['id','text','audio']):
    for row in batch.to_pylist():
        uid=row['id'];speaker,chapter,_=uid.split('-');p=folder/speaker/chapter/(uid+'.flac')
        expected=hashlib.sha256(row['audio']['bytes']).hexdigest();assert sha(p)==expected and references[uid]==row['text'],uid
        info=sf.info(p);assert info.samplerate==16000 and info.channels==1
        record={'id':uid,'audio_path':str(p),'bytes':p.stat().st_size,'sha256':expected,'reference_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'sample_rate':info.samplerate,'channels':info.channels,'audio_seconds':info.duration}
        proof.append(record);items.append({**record,'reference':row['text']})
        if len(items)%500==0:print('TEST_CLEAN_VERIFIED',len(items),flush=True)
assert len(items)==2620 and len({r['id'] for r in items})==2620
items.sort(key=lambda r:r['id']);(ROOT/'data/test-clean-items.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
(ROOT/'results/test-clean-inputs.json').write_text(json.dumps({'passed':True,'count':2620,'source_bytes':350452636,'source_sha256':sha(parquet),'source_revision':'2a4a7918cd87e2f0e315e576b405ae7b7a65d83f','files':sorted(proof,key=lambda r:r['id'])},indent=2)+'\n')
conditions=existing(ROOT/'data/official/formal/conditions/qwen3_asr.jsonl',OLD/'data/official-20261030/formal/conditions/qwen3_asr.jsonl','OFFICIAL_CONDITIONS')
rows=[json.loads(s) for s in conditions.read_text().splitlines()];assert len(rows)==300 and len({r['id'] for r in rows})==300
official=[];metadata=[]
for r in rows:
    params=r['params'];assert r['seed']==42 and params['temperature']==0 and params['max_new_tokens']==1024 and params['language'] is None and params['dtype']=='bf16'
    p=(conditions.parent/r['assets']['audio']['path']).resolve();assert sha(p)==r['assets']['audio']['sha256'],r['id']
    info=sf.info(p);record={'id':r['id'],'source':r['source'],'audio_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'audio_seconds':info.duration,'sample_rate':info.samplerate,'channels':info.channels,'seed':42,'params':params,'condition_sha256':hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()}
    metadata.append(record);official.append({**record,'reference':r['text'],'language_metadata_only':r['language']})
(ROOT/'data/official-items.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in official))
(ROOT/'results/official-inputs.json').write_text(json.dumps({'passed':True,'count':300,'condition_source_path':str(conditions),'condition_source_sha256':sha(conditions),'files':metadata},indent=2)+'\n')
print('ALL_INPUTS_VERIFIED',2620,300,flush=True)
