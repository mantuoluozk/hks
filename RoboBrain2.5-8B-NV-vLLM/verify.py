"""Audit one complete native-vLLM matrix, including backend counter deltas."""
import argparse,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');a=p.parse_args()
O=R/'results'/Path(json.loads((R/'results/formal-current.json').read_text(encoding='utf-8'))['path']).name
required={(n,c) for n in [4096,16384] for c in [1,8,64]};allrows=[]
for repeat in [1,2]:
 rows=[json.loads(line) for line in (O/f'round{repeat}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
 assert len(rows)==6 and {(r['random_input_len'],r['max_concurrency']) for r in rows}==required
 for r in rows:
  count=max(8,2*r['max_concurrency']);assert r['completed']==count and r['random_output_len']==1024
  assert r['total_output_tokens']==count*1024 and len(r['output_lens'])==count and set(r['output_lens'])=={1024}
  assert len(r['errors'])==count and not any(r['errors'])
  assert len(r['generated_texts'])==count and all(t.strip() for t in r['generated_texts']),'Empty visible output needs diagnosis'
  assert len(r['ttfts'])==count and all(t>0 for t in r['ttfts'])
  assert len(r['input_lens'])==count and sum(r['input_lens'])==r['total_input_tokens']
  assert math.isclose(r['output_throughput'],r['total_output_tokens']/r['duration'],rel_tol=1e-6)
  assert math.isclose(r['request_throughput'],count/r['duration'],rel_tol=1e-6)
  assert math.isclose(r['mean_ttft_ms'],1000*sum(r['ttfts'])/count,rel_tol=1e-6)
  for key in ['duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms']:assert math.isfinite(r[key]) and r[key]>0
 allrows.extend(rows)
functional=json.loads((O/'functional/summary.json').read_text(encoding='utf-8'));assert len(functional['cases'])==8 and functional['functional_pass'] and functional['diagnostic_semantic_pass']
assert all(c['usage'] and c['usage']['completion_tokens']>0 for c in functional['cases'])
assert json.loads((O/'weights/weights-check.json').read_text(encoding='utf-8'))['all_required_files_match']
def counter(path,name):
 lines=path.read_text(encoding='utf-8').splitlines();values=[float(l.rsplit(' ',1)[1]) for l in lines if l.startswith(name+'{') or l.startswith(name+' ')];assert values,name;return sum(values)
def delta(name):return counter(O/'metrics-after.prom',name)-counter(O/'metrics-before.prom',name)
count=sum(r['completed'] for r in allrows);tokens=sum(sum(r['output_lens']) for r in allrows)
assert count==608 and tokens==622592
assert delta('vllm:generation_tokens_total')==tokens+12*32
assert delta('vllm:request_success_total')==count+12
report={'model':'RoboBrain2.5-8B-NV-vLLM','job':O.name,'requests':count,'actual_counted_output_tokens':tokens,'functional_cases':8,'backend_generation_delta':delta('vllm:generation_tokens_total'),'backend_success_delta':delta('vllm:request_success_total'),'warmup_requests':12,'warmup_output_tokens_each':32,'complete':True,'independent_process_reproduction':None,'official_acceptance':None,'metrics':[{k:r[k] for k in ['random_input_len','max_concurrency','duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms']} for r in allrows[6:]],'audit_limit':'Throughput and mean TTFT recomputed; per-request full inter-token latency data not retained by client, p99 TPOT is the client result.'}
(O/'verified-summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report,indent=2))
raise SystemExit(0 if a.selfcheck else 2)
