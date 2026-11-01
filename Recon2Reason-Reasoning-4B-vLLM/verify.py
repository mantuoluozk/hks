"""Recompute local evidence; does not start a model or claim official acceptance."""
import json,math,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def equal(a,b):assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-7),(a,b)
def run(folder):
 cached=read(folder/'summary.json');rounds=[];inputs={}
 for ri in (1,2):
  rd=folder/f'round{ri}'
  rows=[json.loads(x) for x in (rd/'requests.jsonl').read_text().splitlines()]
  assert len(rows)==50
  warm=[x for x in rows if x['kind']=='warmup']
  assert [x['request']['messages'][0]['content'].count('Explain the properties') for x in warm]==[32,256]
  measured=[x for x in rows if x['kind']=='measured'];assert len(measured)==48
  stored=read(rd/'summary.json');assert len(stored)==6
  points=[]
  for n in (32,256):
   for c in (1,2,4):
    selected=[x for x in measured if (x['prompt_repeats'],x['concurrency'])==(n,c)]
    assert len(selected)==8
    for x in selected:
     q=x['request'];a=x['response']
     assert q['temperature']==0 and q['max_tokens']==128 and q['ignore_eos'] is True
     assert q['messages'][0]['content']=='Explain the properties of a red rectangle. '*n
     assert a['usage']['completion_tokens']==128 and a['choices'][0]['message']['content'].strip()
     assert a['choices'][0]['finish_reason']=='length'
    lens={x['response']['usage']['prompt_tokens'] for x in selected};assert len(lens)==1
    inputs[str(n)]=list(lens)[0]
    cell=next(x for x in stored if (x['prompt_repeats'],x['concurrency'])==(n,c))
    assert cell['requests']==8 and cell['elapsed']>0
    assert max(x['seconds'] for x in selected)<=cell['elapsed']+0.1
    latency=sum(x['seconds'] for x in selected)/8
    tps=sum(x['response']['usage']['completion_tokens'] for x in selected)/cell['elapsed']
    equal(latency,cell['mean_seconds']);equal(tps,cell['output_tokens_per_second'])
    points.append({**cell,'prompt_tokens':inputs[str(n)]})
  for a,b in zip(cached['rounds'][ri-1],stored):assert a==b
  rounds.append(points)
 deviations=[]
 for a,b in zip(*rounds):
  for metric in ('mean_seconds','output_tokens_per_second'):
   deviations.append(abs(b[metric]-a[metric])/a[metric])
 assert cached['performance_reproduced']==(max(deviations)<=.2)
 for d,a,b in zip(cached['deviations'],*rounds):
  for metric in ('mean_seconds','output_tokens_per_second'):equal(d[metric],abs(b[metric]-a[metric])/a[metric])
 return {'rounds':rounds,'max_deviation_percent':100*max(deviations),'performance_reproduced':max(deviations)<=.2,'measured_requests_per_round':48,'warmups_per_round':2}
def smoke(folder):
 summary=read(folder/'summary.json');assert len(summary['cases'])==5
 answers={}
 for case in summary['cases']:
  label=case['case'];raw=(folder/f'{label}.response.txt').read_text()
  if label=='stream_image':
   events=[x[6:] for x in raw.splitlines() if x.startswith('data: ')]
   assert '[DONE]' in events
   data=[json.loads(x) for x in events if x!='[DONE]']
   answer=''.join(x['choices'][0]['delta'].get('content') or '' for x in data if x.get('choices'))
  else:answer=json.loads(raw)['choices'][0]['message']['content']
  assert answer==case['answer'] and answer.strip()
  answers[label]=answer.strip().lower()
 assert re.fullmatch(r'5[.!]?',answers['text'])
 assert re.search(r'\bred\b',answers['single_image'])
 assert re.findall(r'\b(?:red|blue)\b',answers['multi_image'])[:2]==['red','blue']
 assert re.search(r'\bblue\b',answers['multi_turn']) and re.search(r'\bred\b',answers['stream_image'])
 return {'path':folder.name,'checks':5,'passed':True}
def main():
 old=run(ROOT/'results/validation-20261031T153511Z')
 fresh=run(ROOT/'results/validation-20261031T160644Z')
 cross=[]
 for a,b in zip(old['rounds'][0],fresh['rounds'][0]):
  for metric in ('mean_seconds','output_tokens_per_second'):cross.append(abs(b[metric]-a[metric])/a[metric])
 cross_all=[abs(b[metric]-a[metric])/a[metric] for new_round in fresh['rounds'] for a,b in zip(old['rounds'][0],new_round) for metric in ('mean_seconds','output_tokens_per_second')]
 diagnostic_names=['smoke-20261031T153901Z','smoke-20261031T154453Z','smoke-20261031T160954Z','smoke-20261031T161544Z']
 result={'model':'Recon2Reason-Reasoning-4B','framework':'vLLM','old_image':old,'dedicated_image':fresh,'cross_image_first_round_max_deviation_percent':100*max(cross),'cross_image_both_rounds_vs_original_first_round_max_percent':100*max(cross_all),'smoke':[smoke(ROOT/'results'/name) for name in diagnostic_names],'official_accuracy_required':False,'official_acceptance':None,'performance_scope':'synthetic text-only HTTP requests; multimodal functionality checked separately'}
 assert len(result['smoke'])==4
 assert fresh['performance_reproduced'] and max(cross)<=.2
 if '--diagnostic-only' not in sys.argv:
  completed=subprocess.run([sys.executable,str(ROOT/'verify_formal.py')],capture_output=True,text=True)
  if completed.returncode:
   print(completed.stdout);print(completed.stderr,file=sys.stderr)
   raise SystemExit(completed.returncode)
  current=json.loads((ROOT/'results/formal-current.json').read_text())
  formal=read(ROOT/'results'/Path(current['path']).name/'verified-summary.json')
  result['formal_matrix']=formal
  result['performance_scope']='Official-template image matrix; text-only diagnostics are recorded separately'
  result['performance_reproduced']=formal['performance_reproduced']
  (ROOT/'verify_result.json').write_text(json.dumps(result,indent=2))
  print(json.dumps({'model':result['model'],'formal_requests':1216,'formal_template_metrics':9,'formal_max_deviation_percent':formal['max_deviation_percent'],'confirmed_special_only_requests':len(formal['confirmed_special_only_requests']),'performance_reproduced':formal['performance_reproduced'],'official_acceptance':None},indent=2))
 else:
  (ROOT/'results/diagnostic-verified.json').write_text(json.dumps(result,indent=2))
  print('Text-only diagnostic check passed; max deviation percent:',fresh['max_deviation_percent'])
if __name__=='__main__':main()
