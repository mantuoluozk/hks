"""Independently recompute the single-pass eager/graph short comparison."""
import argparse,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
 job=ROOT/'results'/Path(a.job).name;results=[];transcripts=[]
 for name in ('eager','decode-graph'):
  rd=job/name;rows=list(map(json.loads,(rd/'requests.jsonl').read_text(encoding='utf-8').splitlines()))
  assert len(rows)==50 and sum(x['kind']=='warmup' for x in rows)==2
  transcripts.append(rows);cells=json.loads((rd/'summary.json').read_text(encoding='utf-8'));assert len(cells)==6
  for cell in cells:
   n,c=cell['prompt_repeats'],cell['concurrency']
   selected=[x for x in rows if x['kind']=='measured' and (x['prompt_repeats'],x['concurrency'])==(n,c)]
   assert len(selected)==cell['requests']==8
   for r in selected:
    q,resp=r['request'],r['response']
    assert q['temperature']==0 and q['ignore_eos'] is True and q['max_tokens']==128
    assert q['messages'][0]['content']=='Explain the properties of a red rectangle. '*n
    assert resp['usage']['completion_tokens']==128 and resp['choices'][0]['finish_reason']=='length'
    assert resp['choices'][0]['message']['content'].strip()
   assert max(x['seconds'] for x in selected)<=cell['elapsed']+.1
   tps=sum(x['response']['usage']['completion_tokens'] for x in selected)/cell['elapsed']
   assert math.isclose(tps,cell['output_tokens_per_second'],rel_tol=1e-9)
  results.append(cells)
 equality=[]
 for x,y in zip(*transcripts):
  equality.append({'request':x['request']==y['request'],'usage':x['response']['usage']==y['response']['usage'],'text':x['response']['choices'][0]['message']['content']==y['response']['choices'][0]['message']['content']})
 comparison=[]
 for x,y in zip(*results):
  assert (x['prompt_repeats'],x['concurrency'])==(y['prompt_repeats'],y['concurrency'])
  comparison.append({'prompt_repeats':x['prompt_repeats'],'concurrency':x['concurrency'],'gain_percent':100*(y['output_tokens_per_second']/x['output_tokens_per_second']-1)})
 report={'cells':comparison,'measured_requests_per_configuration':48,'warmups_per_configuration':2,'single_pass':True,'paired_responses':equality,'all_requests_usage_text_equal':all(all(v.values()) for v in equality),'official_acceptance':None}
 (job/'independent-screen-audit.json').write_text(json.dumps(report,indent=2),newline="\n");print(json.dumps(report,indent=2))
if __name__=='__main__':main()
