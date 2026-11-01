"""Check complete official-template matrix and independent-process reproduction."""
import argparse,json,math,shutil
from verify import smoke
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def close(a,b):assert math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6),(a,b)
def confirmed_special_only():
 folder=ROOT/'results/empty-output-diagnosis'
 config=json.loads((ROOT/'sources/publisher-tokenizer-config.json').read_text())
 token=151673
 spec=config['added_tokens_decoder'][str(token)]
 assert spec['special'] is True and spec['content']=='<|feat_pad|>'
 references={r['index']:r for r in json.loads((folder/'publisher-reference.json').read_text())}
 confirmed={}
 for index in (71,98):
  visible=json.loads((folder/f'full-index{index}-skipFalse.response.json').read_text())
  hidden=json.loads((folder/f'full-index{index}-skipTrue.response.json').read_text())
  ids=visible['choices'][0]['token_ids']
  assert ids==hidden['choices'][0]['token_ids']==[token]*1024
  assert visible['choices'][0]['message']['content']==spec['content']*1024
  assert not hidden['choices'][0]['message']['content']
  for response in (visible,hidden):
   assert response['usage']['completion_tokens']==1024
   assert response['choices'][0]['finish_reason']=='length'
  ref=references[index]
  assert ref['new_tokens']==ids[:64] and ref['content']==spec['content']*64
  assert ref['input_tokens']==visible['usage']['prompt_tokens']==hidden['usage']['prompt_tokens']
  confirmed[index]=ref['input_tokens']
 return confirmed
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job');args=parser.parse_args()
 name=Path(args.job or json.loads((ROOT/'results/formal-current.json').read_text())['path']).name
 job=ROOT/'results'/name
 confirmed=confirmed_special_only()
 required={(n,1024,c) for n in (4096,16384) for c in (1,8,64)}
 datasets=[]
 smoke_checks=[]
 special_cases=[]
 for process in (1,2):
  smoke_log=(job/f'process{process}/smoke.log').read_text()
  path=next(x.split('Artifacts: ',1)[1].strip() for x in smoke_log.splitlines() if x.startswith('Artifacts: '))
  smoke_checks.append(smoke(ROOT/'results'/Path(path).name))
  repeats=[]
  for repeat in (1,2):
   rows=[json.loads(x) for x in (job/f'process{process}/round{repeat}.jsonl').read_text().splitlines()]
   assert len(rows)==6
   cells={}
   for row in rows:
    key=(row['random_input_len'],row['random_output_len'],row['max_concurrency'])
    assert key not in cells
    count=max(8,key[2]*2)
    assert row['completed']==count and row['total_output_tokens']==count*1024
    assert len(row['output_lens'])==count and set(row['output_lens'])=={1024}
    assert len(row['errors'])==count and not any(row['errors'])
    assert len(row['generated_texts'])==count
    empty=[i for i,t in enumerate(row['generated_texts']) if not t.strip()]
    allowed=key==(16384,1024,64)
    assert set(empty).issubset(set(confirmed)) if allowed else not empty
    for i in empty:
     assert row['input_lens'][i]==confirmed[i]
     assert row['generated_texts'][i]=='' and row['ttfts'][i]==0 and row['itls'][i]==[]
     special_cases.append({'process':process,'repeat':repeat,'index':i,'output_tokens':1024,'token_id':151673})
    assert len(row['input_lens'])==count and all(x>0 for x in row['input_lens'])
    assert len(row['ttfts'])==count and all(x>0 or i in empty for i,x in enumerate(row['ttfts']))
    close(row['output_throughput'],row['total_output_tokens']/row['duration'])
    close(row['request_throughput'],count/row['duration'])
    close(row['mean_ttft_ms'],1000*sum(row['ttfts'])/count)
    for field in ('output_throughput','mean_ttft_ms','mean_tpot_ms','mean_e2e_latency_ms'):
     assert math.isfinite(row[field]) and row[field]>0
    cells[key]=row
   assert set(cells)==required
   repeats.append(cells)
  datasets.append(repeats)
 comparisons=[]
 for key in sorted(required):
  a,b=datasets[0][1][key],datasets[1][1][key]
  comparisons.append({'input':key[0],'output':key[1],'concurrency':key[2],
    'deviation_percent':{m:100*abs(b[m]-a[m])/abs(a[m]) for m in ('duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms')}})
 worst=max(v for row in comparisons for v in row['deviation_percent'].values())
 report={'job':name,'template':'image, 4096/16384 text length parameter, 1024 output, concurrency 1/8/64; second repetition measured',
   'processes':2,'requests_per_process_excluding_client_internal_warmup':608,'complete':True,
   'smoke':smoke_checks,'confirmed_special_only_requests':special_cases,'no_visible_token_ttft_count':len(special_cases),
   'ttft_interpretation':'Raw client statistics retained; the eight hidden-special-only requests have no visible-content TTFT and are represented as zero by the unmodified client. No requests removed.',
   'comparison':comparisons,'max_deviation_percent':worst,'performance_reproduced':worst<=20,'official_acceptance':None}
 (job/'verified-summary.json').write_text(json.dumps(report,indent=2))
 bench=ROOT/'bench';bench.mkdir(exist_ok=True)
 (bench/'reproduce').mkdir(exist_ok=True)
 for repeat in (1,2):
  shutil.copyfile(job/f'process1/round{repeat}.jsonl',bench/f'round{repeat}.jsonl')
  shutil.copyfile(job/f'process2/round{repeat}.jsonl',bench/f'reproduce/round{repeat}.jsonl')
 (bench/'comparison.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 if worst>20:raise SystemExit(2)
if __name__=='__main__':main()
