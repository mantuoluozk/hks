"""Validate all 12 fixed-length benchmark workloads and their runtime snapshots."""
import json,sys,math,os
from pathlib import Path

root=Path(sys.argv[1])
expected={(i,1024,c):max(8,2*c) for i in (4096,16384) for c in (1,8,64)}
checked=[]
for round_id in (1,2):
    rows=[json.loads(line) for line in (root/f'round{round_id}.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows)==6, (round_id,len(rows))
    seen=set()
    for r in rows:
        key=(r['random_input_len'],r['random_output_len'],r['max_concurrency'])
        assert key in expected and key not in seen,key
        seen.add(key);count=expected[key]
        assert r['completed']==count,(key,r['completed'])
        assert r['total_output_tokens']==count*1024,(key,r['total_output_tokens'])
        info=r['server_info']
        assert info['dtype']=='bfloat16' and info['tp_size']==1
        assert info['max_running_requests']==64
        assert info['attention_backend']=='triton' and info['mm_attention_backend']=='sdpa'
        assert info['enable_deterministic_inference'] is True and info['enable_multimodal'] is True
        assert info['cuda_graph_backend_decode']=='full' and info['cuda_graph_max_bs_decode']==32
        assert r['dataset_name']=='image' and r['total_input_tokens']>count*key[0]
        assert all(math.isfinite(r[k]) and r[k]>0 for k in ['output_throughput','mean_ttft_ms','mean_tpot_ms'])
        log=root/f'round{round_id}_{key[0]}_1024_c{key[2]}.log'
        assert 'Validated completed requests and fixed output token count' in log.read_text(encoding='utf-8')
        checked.append({'round':round_id,'input':key[0],'concurrency':key[2],'completed':count,
                        'generated_tokens':r['total_output_tokens'],'actual_input_tokens':r['total_input_tokens'],
                        'retokenized_tokens':r.get('total_output_tokens_retokenized')})
report={'complete':True,'workloads':checked,'official_scnet_validation':False,'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned'),
        'token_note':'Input settings describe text plus a separate 1080p image; actual input totals include vision/template tokens. Fixed decode length uses server generated tokens; re-tokenized visible output is recorded separately.'}
(root/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
