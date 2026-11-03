"""Independent audio timing/statistics reconstruction; no model calls."""
import argparse,base64,hashlib,json,math,statistics,shutil
from pathlib import Path
from evidence_io import read_json,read_jsonl
ROOT=Path(__file__).resolve().parents[1]
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),(a,b)
def percentile(values,q):
    v=sorted(values);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def check(folder,short=False):
    summary=read_json(folder/'summary.json');rows=read_jsonl(folder/'requests.jsonl');warm=read_jsonl(folder/'warmups.jsonl')
    expected=8 if short else 128;assert summary['passed'] and summary['counted_requests']==len(rows)==expected
    assert len({r['response']['id'] for r in rows})==len(rows)
    assert summary['warmup_requests']==len(warm)==(4 if short else 16)
    fixtures={r['sha256']:r for r in read_json(ROOT/'results/audio-properties.json')['files']}
    for r in rows+warm:
        assert r['success'] and not r['error'] and r['text'] and r['finish_reason']!='length'
        assert r['usage']['completion_tokens']>0 and r['usage']['completion_tokens']<=1024 and r['usage']['prompt_tokens']>0
        assert r['text']==r['response']['choices'][0]['message']['content'].split('<asr_text>',1)[-1].strip()
        assert r['request']['reference_or_language_supplied'] is False
        close(r['audio_seconds'],fixtures[r['sha256']]['duration']);close(r['rtf'],r['elapsed_seconds']/r['audio_seconds'])
        assert r['elapsed_seconds']>0
    expected_cells={(n,c) for n in ([1] if short else [1,2]) for c in ([1,4] if short else [1,4,8,16])}
    assert {(r['round'],r['concurrency']) for r in summary['cells']}==expected_cells
    for cell in summary['cells']:
        n,c=cell['round'],cell['concurrency'];batch=[r for r in rows if (r['round'],r['concurrency'])==(n,c)]
        assert cell['requests']==len(batch)==(4 if short else max(8,2*c))
        assert len([r for r in warm if (r['round'],r['concurrency'])==(n,c)])==2
        total=sum(r['audio_seconds'] for r in batch);lat=[r['elapsed_seconds'] for r in batch]
        close(cell['audio_seconds'],total);close(cell['audio_seconds_per_wall_second'],total/cell['wall_seconds'])
        close(cell['mean_request_rtf'],statistics.mean(r['rtf'] for r in batch));close(cell['mean_latency_seconds'],statistics.mean(lat))
        close(cell['max_latency_seconds'],max(lat));close(cell['p95_latency_seconds'],percentile(lat,.95));close(cell['p99_latency_seconds'],percentile(lat,.99))
        assert cell['completion_tokens']==sum(r['usage']['completion_tokens'] for r in batch) and cell['errors']==0
    return {'passed':True,'counted_requests':expected,'warmup_requests':len(warm),'cells_checked':len(summary['cells']),'model_requests_sent':0}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--short',action='store_true');p.add_argument('--publish-index',action='store_true');a=p.parse_args()
    a.folder=a.folder.resolve();assert a.folder.is_relative_to(ROOT)
    result=check(a.folder,a.short)
    warm=read_jsonl(a.folder/'warmups.jsonl')
    for r in warm:
        ref=r['request'];blob=Path(ref['body_path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==ref['body_sha256'];body=json.loads(blob)
        assert body['temperature']==0 and body['max_tokens']==1024 and len(body['messages'][0]['content'])==1 and body['messages'][0]['content'][0]['type']=='audio_url' and 'language' not in body
        url=body['messages'][0]['content'][0]['audio_url']['url'];assert hashlib.sha256(base64.b64decode(url.split(',',1)[1])).hexdigest()==r['sha256']
    result['warm_wire_bodies_checked']=len(warm)
    (a.folder/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.publish_index:
        dest=ROOT/'bench';dest.mkdir(exist_ok=True)
        for name in ['round1.jsonl','round2.jsonl','summary.json','requests.jsonl','warmups.jsonl','independent-check.json']:shutil.copyfile(a.folder/name,dest/name)
        (dest/'evidence-index.json').write_text(json.dumps({'source':str(a.folder.relative_to(ROOT)),'timing':'Audio end-to-end; natural EOS; no forced token count'},indent=2)+'\n')
    print(json.dumps(result))
