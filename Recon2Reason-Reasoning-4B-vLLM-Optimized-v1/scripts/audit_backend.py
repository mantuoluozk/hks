"""Audit the saved pre-diagnostic server counters against the fixed workload."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
 job=ROOT/'results'/Path(a.job).name
 rows=[]
 for repeat in (1,2):rows.extend(map(json.loads,(job/f'process1/round{repeat}.jsonl').read_text(encoding='utf-8').splitlines()))
 assert len(rows)==12 and sum(r['completed'] for r in rows)==608
 matrix_tokens=sum(sum(r['output_lens']) for r in rows);assert matrix_tokens==622592
 log=(job/'process1/smoke.log').read_text(encoding='utf-8')
 name=next(l.split('Artifacts: ',1)[1] for l in log.splitlines() if l.startswith('Artifacts: '))
 smoke=ROOT/'results'/Path(name).name
 cases=json.loads((smoke/'summary.json').read_text(encoding='utf-8'))['cases'];assert len(cases)==5
 nonstream_tokens=sum(r['usage']['completion_tokens'] for r in cases if r['usage'] is not None)
 stream=next(r for r in cases if r['case']=='stream_image');assert stream['answer']=='red'
 # The fixed smoke response has one visible red token and a stop token. Save this
 # explicit assumption alongside exact observed counters, rather than retokenizing
 # arbitrary generated benchmark text as a replacement for real usage.
 stream_tokens=2
 metrics=(job/'process1/server-metrics.prom').read_text(encoding='utf-8')
 def counter(name):
  values=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith(name+'{') or l.startswith(name+' ')]
  assert values,name
  return sum(values)
 actual=counter('vllm:generation_tokens_total');success=counter('vllm:request_success_total')
 expected=matrix_tokens+12*32+nonstream_tokens+stream_tokens
 assert actual==expected,(actual,expected)
 assert success==608+12+5,success
 report={'matrix_requests':608,'matrix_output_tokens':matrix_tokens,'internal_warmups':12,'warmup_output_tokens_each':32,'nonstream_functional_output_tokens':nonstream_tokens,'stream_functional_expected_tokens':stream_tokens,'stream_assumption':'Fixed red response: one visible token plus stop token; exact saved backend total matches this accounting.','expected_generation_tokens':expected,'observed_generation_tokens':actual,'successful_requests':success,'captured_before_special_token_diagnostics':True,'official_acceptance':None}
 (job/'backend-audit.json').write_text(json.dumps(report,indent=2),newline="\n");print(json.dumps(report,indent=2))
if __name__=='__main__':main()
