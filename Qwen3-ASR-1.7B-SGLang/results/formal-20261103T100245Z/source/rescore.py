"""Re-score saved raw responses against verified input references; no model calls."""
import argparse,csv,hashlib,json
from pathlib import Path
from asr_api import transcript
from score_asr import score,summary
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['accuracy','official'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
name='test-clean-items.jsonl' if a.stage=='accuracy' else 'official-items.jsonl'
items=[json.loads(s) for s in (ROOT/'data'/name).read_text().splitlines()];index={r['id']:r for r in items}
rows=[json.loads(s) for s in (a.output/'predictions.jsonl').read_text().splitlines()];assert len(rows)==len(items) and {r['id'] for r in rows}==set(index)
for r in rows:
    item=index[r['id']];assert r['sha256']==item['sha256']
    request=r.get('request')
    if request:
        blob=Path(request['body_path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==request['body_sha256'];body=json.loads(blob)
        assert len(body['messages'])==1 and len(body['messages'][0]['content'])==1 and body['messages'][0]['content'][0]['type']=='audio_url'
        assert body['temperature']==0 and body['max_tokens']==1024 and 'language' not in body
    text=transcript(r['response']['choices'][0]['message'].get('content') or '') if r['success'] else ''
    assert text==r['text']
    if a.stage=='accuracy':
        expected=score(item['reference'],text)
        for key,value in expected.items():assert r[key]==value,(r['id'],key)
    else:
        assert r['condition_sha256']==item['condition_sha256'] and body['seed']==42
        assert (a.output/'outputs'/(r['id']+'.txt')).read_text(encoding='utf-8')==text+'\n'
if a.stage=='accuracy':
    result=summary(rows);cached=json.loads((a.output/'summary.json').read_text());assert result=={k:v for k,v in cached.items() if k!='wall_seconds'}
else:
    assert [r['id'] for r in rows]==[r['id'] for r in items]
    with (a.output/'反馈表.csv').open(encoding='utf-8-sig',newline='') as f:feedback=list(csv.DictReader(f))
    assert [r['condition_id'] for r in feedback]==[r['id'] for r in items] and all(r['status']=='成功' and r['parameter_changes']=='无' for r in feedback)
(a.output/'rescore.json').write_text(json.dumps({'passed':True,'rows_checked':len(rows),'raw_responses_and_input_hashes_checked':True,'reference_and_language_not_sent':True,'model_requests_sent':0},indent=2)+'\n');print('INDEPENDENT_RECHECK',a.stage,len(rows),flush=True)
