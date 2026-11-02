"""Recompute saved graph matrix and compare every template metric to original v1."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from verify import smoke
ROOT=Path(__file__).resolve().parent
METRICS=('duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6),(a,b)
def percentile(v,p):
 s=sorted(v);x=(len(s)-1)*p;i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
 job=ROOT/'results'/Path(args.job).name
 required={(n,c) for n in (4096,16384) for c in (1,8,64)}
 evidence=json.loads((job/'graph-empty-output/summary.json').read_text(encoding='utf-8'))
 server_log=(job/'process1/server.log').read_text(encoding='utf-8')
 assert 'enforce_eager=False' in server_log and 'FULL_DECODE_ONLY' in server_log
 for e in evidence:
  assert len(e['responses'])==2
  for skip,r in zip((False,True),e['responses']):
   f=job/'graph-empty-output'/r['path'];raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
   a=json.loads(raw);ch=a['choices'][0];assert ch['token_ids']==[151673]*1024
   request=f.with_name(f.name.replace('.response.json','.request.json')).read_bytes()
   assert hashlib.sha256(request).hexdigest()==r['request_sha256']
   reference=json.loads((ROOT/'sources/special-probe-request-hashes.json').read_text(encoding='utf-8'))
   assert r['request_sha256']==reference[f"full-index{e['index']}-skip{skip}.json"]
   assert a['usage']['prompt_tokens']==e['input_tokens'] and a['usage']['completion_tokens']==1024
   assert ch['finish_reason']=='length'
   assert ch['message']['content']==('' if skip else '<|feat_pad|>'*1024)
 rounds=[];empty=[]
 for repeat in (1,2):
  cells={}
  for r in map(json.loads,(job/f'process1/round{repeat}.jsonl').read_text(encoding='utf-8').splitlines()):
   key=(r['random_input_len'],r['max_concurrency']);assert key not in cells
   count=max(8,2*key[1]);assert r['random_output_len']==1024 and r['completed']==count
   for field in ('output_lens','input_lens','errors','ttfts','itls','generated_texts'):assert len(r[field])==count
   assert not any(r['errors']) and set(r['output_lens'])=={1024}
   assert r['total_output_tokens']==sum(r['output_lens'])==1024*count
   assert r['total_input_tokens']==sum(r['input_lens'])
   for i,text in enumerate(r['generated_texts']):
    if not text.strip():
     assert key==(16384,64)
     proof=next(e for e in evidence if e['repeat']==repeat and e['index']==i)
     assert proof['input_tokens']==r['input_lens'][i] and r['ttfts'][i]==0 and r['itls'][i]==[]
     empty.append({'repeat':repeat,'index':i})
    else:assert r['ttfts'][i]>0
   close(r['request_throughput'],count/r['duration'])
   close(r['output_throughput'],sum(r['output_lens'])/r['duration'])
   close(r['total_throughput'],(sum(r['input_lens'])+sum(r['output_lens']))/r['duration'])
   close(r['mean_ttft_ms'],statistics.mean(r['ttfts'])*1000)
   close(r['p99_ttft_ms'],percentile(r['ttfts'],.99)*1000)
   for m in METRICS:assert math.isfinite(r[m]) and r[m]>0
   cells[key]=r
  assert set(cells)==required;rounds.append(cells)
 assert len(empty)==len(evidence)
 log=(job/'process1/smoke.log').read_text(encoding='utf-8');name=next(l.split('Artifacts: ',1)[1] for l in log.splitlines() if l.startswith('Artifacts: '))
 functional=smoke(ROOT/'results'/Path(name).name)
 baseline={ (r['random_input_len'],r['max_concurrency']):r for r in map(json.loads,(ROOT/'results/formal-20261031T163337Z/process1/round2.jsonl').read_text(encoding='utf-8').splitlines()) }
 comparison=[]
 for key in sorted(required):
  a,b=baseline[key],rounds[1][key];assert a['input_lens']==b['input_lens'] and a['output_lens']==b['output_lens']
  comparison.append({'input':key[0],'concurrency':key[1],'baseline':{m:a[m] for m in METRICS},'optimized':{m:b[m] for m in METRICS},'change_percent':{m:100*(b[m]/a[m]-1) for m in METRICS}})
 result={'complete':True,'processes':1,'matrix_requests':608,'comparison':comparison,'functional':functional,'hidden_special_requests':empty,'independent_process_reproduction':None,'official_acceptance':None,'audit_scope':'Counts, input lengths, throughput, mean/p99 TTFT recomputed. TPOT/E2E retained from stock client; individual E2E latencies are absent from raw JSONL.'}
 (job/'verified-summary.json').write_text(json.dumps(result,indent=2),newline="\n");print(json.dumps(result,indent=2))
if __name__=='__main__':main()
