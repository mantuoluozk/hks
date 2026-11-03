"""Recompute a completed run and publish relative evidence indexes; no model calls."""
import argparse,json,re,shutil
from pathlib import Path
from check_cell import check
from evidence_codec import read_text

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--job',type=Path);a=p.parse_args()
job=a.job or Path(json.loads((ROOT/'results/formal-current.json').read_text())['path'])
job=job.resolve();assert job.is_relative_to((ROOT/'results').resolve())
state=json.loads((job/'state.json').read_text());assert state['stage']=='completed'
count=0;scored=[]
for repeat in [1,2]:
    rows=[json.loads(x) for x in read_text(job/f'round{repeat}.jsonl').splitlines()]
    assert len(rows)==6
    assert {(r['random_input_len'],r['max_concurrency']) for r in rows}=={(n,c) for n in [4096,16384] for c in [1,8,64]}
    for row in rows:
        length=row['random_input_len'];c=row['max_concurrency']
        count+=check(row,job/f'round{repeat}_{length}_1024_c{c}-raw',length,c)
    if repeat==2:scored=rows
assert count==608
def metric(path,name):
    values=re.findall('^'+re.escape(name)+r'(?:\{[^}]*\})?\s+([0-9.eE+\-]+)',path.read_text(),re.M)
    return sum(map(float,values)) if values else None
counter={}
for name in ['vllm:request_success_total','vllm:generation_tokens_total','vllm:request_generation_tokens_sum','vllm:prompt_tokens_total','vllm:request_prompt_tokens_sum']:
    before=metric(job/'before-matrix.prom',name);after=metric(job/'after-matrix.prom',name)
    counter[name]=None if before is None or after is None else after-before
if counter['vllm:request_success_total'] is not None:assert counter['vllm:request_success_total']==620,counter
if counter['vllm:request_generation_tokens_sum'] is not None:assert counter['vllm:request_generation_tokens_sum']==622976,counter
if counter['vllm:request_prompt_tokens_sum'] is not None:assert counter['vllm:request_prompt_tokens_sum']==6348800,counter
kv=re.findall(r'GPU KV cache size: ([0-9,]+) tokens',(job/'server.log').read_text())
summary={'complete':True,'job':job.name,'counted_requests':count,'counted_output_tokens':count*1024,
         'scored_requests':304,'client_warmup_requests':12,'client_warmup_output_tokens':384,
         'server_counter_delta':counter,'kv_capacity_tokens':int(kv[-1].replace(',','')) if kv else None,
         'client_concurrency':[1,8,64],'max_num_seqs':64,'round2':scored,
         'online_accuracy_executed':False,'independent_service_full_reproduction':None,'official_acceptance':None}
(job/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
bench=ROOT/'bench';bench.mkdir(exist_ok=True)
for name in ['round1.jsonl','round2.jsonl','summary.json']:shutil.copyfile(job/name,bench/name)
(bench/'evidence-index.json').write_text(json.dumps({'job':'../results/'+job.name,
    'original_requests':'../results/'+job.name+'/round*-raw/requests-and-timing.jsonl',
    'original_inputs':'../results/'+job.name+'/round*-raw/inputs.json',
    'timing_scope':'Native token-ID /v1/completions client end-to-end; includes server queueing'},indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='round2'}))
