"""Recompute one native-token benchmark cell from per-request timings."""
import argparse,json,math
from pathlib import Path

def check(row,raw,length,c):
    n=max(8,2*c)
    assert row['random_input_len']==length and row['random_output_len']==1024 and row['max_concurrency']==c
    assert row['completed']==n and row['total_output_tokens']==n*1024
    assert row['input_lens']==[length]*n and row['output_lens']==[1024]*n
    assert len(row['errors'])==n and not any(row['errors'])
    assert len(row['generated_texts'])==n and all(x.strip() for x in row['generated_texts'])
    records=[json.loads(x) for x in (raw/'requests-and-timing.jsonl').read_text(encoding='utf-8').splitlines()]
    measured=[r for r in records if r['request']['output_len']==1024]
    warm=[r for r in records if r['request']['output_len']==32]
    assert len(measured)==n and len(warm)==1
    for r in records:
        q,a=r['request'],r['response'];assert a['success'] and not a['error']
        assert a['output_len']==q['output_len'] and a['generated_text'].strip()
        assert q['prompt_len']==length and len(q['prompt'])==length
        assert math.isfinite(a['latency']) and a['latency']>=a['ttft']>0
    def close(a,b):assert math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6),(a,b)
    close(row['output_throughput'],n*1024/row['duration'])
    close(row['mean_ttft_ms'],1000*sum(r['response']['ttft'] for r in measured)/n)
    close(row['mean_e2e_latency_ms'],1000*sum(r['response']['latency'] for r in measured)/n)
    close(row['mean_tpot_ms'],1000*sum((r['response']['latency']-r['response']['ttft'])/1023 for r in measured)/n)
    return n

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--file',type=Path,required=True);p.add_argument('--input',type=int,required=True);p.add_argument('--concurrency',type=int,required=True);p.add_argument('--raw',type=Path,required=True);a=p.parse_args()
    row=json.loads(a.file.read_text(encoding='utf-8').splitlines()[-1]);print('VERIFIED_CELL_REQUESTS',check(row,a.raw,a.input,a.concurrency))
