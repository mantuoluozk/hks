"""Recalculate recorded HTTP functional and performance results without inference."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'results/evidence-manifest.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text())['files']:
            target=(ROOT/item['path']).resolve()
            assert target.is_relative_to(ROOT)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],item['path']
    sets=[]
    for name in ['http-v5','http-v6']:
        base=ROOT/'results'/name
        checks=json.loads((base/'checks/requests.json').read_text())
        assert len(checks)==12 and all(x['status']==x['expected_status'] for x in checks)
        valid={x['name']:x['response'] for x in checks if x['status']==200}
        assert valid['health']['ready']
        assert valid['red']['output_ids']==valid['red-repeat']['output_ids']
        reference=json.loads((ROOT/'results/reference-probe-v5/summary.json').read_text())
        def strip(ids):return ids[:-1] if ids and ids[-1] in [128001,128009] else ids
        for expected in reference['cases']:
            assert strip(valid[expected['case']]['output_ids'])==strip(expected['output_ids'])
        for count in [2,8]:assert valid[f'images-{count}']['images']==count
        rows=[json.loads(x) for x in (base/'performance/requests.jsonl').read_text().splitlines()]
        summary=json.loads((base/'performance/summary.json').read_text())
        assert len(rows)==12
        for item in summary['results']:
            selected=[r for r in rows if r['images']==item['images']]
            assert [r['repeat'] for r in selected]==[0,1,2,3]
            assert [r['warmup'] for r in selected]==[True,False,False,False]
            counted=selected[1:];elapsed=sum(r['seconds'] for r in counted)
            tokens=sum(r['response']['output_tokens'] for r in counted)
            assert all(len(r['response']['output_ids'])==r['response']['output_tokens'] for r in selected)
            assert abs(elapsed/3-item['mean_seconds'])<1e-9
            assert abs(tokens/elapsed-item['output_tokens_per_second'])<1e-9
        sets.append(rows)
    comparison=json.loads((ROOT/'results/http-reproducibility.json').read_text())
    for old,new in zip(*sets):assert old['request_sha256']==new['request_sha256']
    for entry in comparison['results']:
        a=statistics.mean(x['seconds'] for x in sets[0] if x['images']==entry['images'] and not x['warmup'])
        b=statistics.mean(x['seconds'] for x in sets[1] if x['images']==entry['images'] and not x['warmup'])
        assert abs(entry['deviation_percent']-abs(b-a)/a*100)<1e-9
    passed=all(x['deviation_percent']<=20 for x in comparison['results'])
    assert passed==comparison['within_20_percent']
    result={'evidence_consistent':True,'functional_checks':24,'performance_requests':24,
            'performance_within_20_percent':passed,'official_accuracy_tested':False}
    print(json.dumps(result))
    (ROOT/'results/selfcheck.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
