"""Check complete official-template matrix and independent-process reproduction."""
import argparse,json,math,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def close(a,b):assert math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6),(a,b)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job');parser.add_argument('--processes',type=int,choices=(1,2),default=2);args=parser.parse_args()
 name=Path(args.job or json.loads((ROOT/'results/formal-current.json').read_text())['path']).name
 job=ROOT/'results'/name

 required={(n,1024,c) for n in (4096,16384) for c in (1,8,64)}
 datasets=[]
 smoke_checks=[]
 special_cases=[]
 for process in range(1,args.processes+1):
  functional=json.loads((job/f'process{process}/functional/summary.json').read_text())
  assert functional['all_passed'] and functional['functional_passed']==functional['functional_total']==8
  smoke_checks.append(functional)
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
    assert not empty,('Empty visible benchmark output requires diagnosis',process,repeat,key,empty)
    assert len(row['input_lens'])==count and all(x>0 for x in row['input_lens'])
    assert len(row['ttfts'])==count and all(x>0 for x in row['ttfts'])
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
 for key in (sorted(required) if args.processes==2 else []):
  a,b=datasets[0][1][key],datasets[1][1][key]
  comparisons.append({'input':key[0],'output':key[1],'concurrency':key[2],
    'deviation_percent':{m:100*abs(b[m]-a[m])/abs(a[m]) for m in ('duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms')}})
 worst=max((v for row in comparisons for v in row['deviation_percent'].values()),default=None)
 report={'job':name,'template':'image, 4096/16384 text length parameter, 1024 output, concurrency 1/8/64; second repetition measured',
   'processes':args.processes,'requests_per_process_excluding_client_internal_warmup':608,'complete':True,
   'smoke':smoke_checks,'confirmed_special_only_requests':special_cases,'no_visible_token_ttft_count':len(special_cases),
   'ttft_interpretation':'First nonempty streamed content; raw client statistics retained.',
   'comparison':comparisons,'max_deviation_percent':worst,'performance_reproduced':(worst<=20 if worst is not None else None),'official_acceptance':None}
 (job/'verified-summary.json').write_text(json.dumps(report,indent=2))
 bench=ROOT/'bench';bench.mkdir(exist_ok=True)
 (bench/'reproduce').mkdir(exist_ok=True)
 for repeat in (1,2):
  shutil.copyfile(job/f'process1/round{repeat}.jsonl',bench/f'round{repeat}.jsonl')
  if args.processes==2:shutil.copyfile(job/f'process2/round{repeat}.jsonl',bench/f'reproduce/round{repeat}.jsonl')
 (bench/'comparison.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 if worst is not None and worst>20:raise SystemExit(2)
if __name__=='__main__':main()
