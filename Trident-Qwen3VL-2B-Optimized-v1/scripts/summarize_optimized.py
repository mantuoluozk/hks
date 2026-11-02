"""Audit returned tokens, vectors, timing and historical baseline comparison."""
import json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
job=R/'results'/Path(read(R/'results/final-current.json')['path']).name
assert read(job/'state.json')['status']=='completed'
data=read(job/'performance/summary.json')['cells']
expected={(n,k,b) for n in [1,2] for k in ['text','image','fused'] for b in [1,2,4,8]}
assert len(data)==24 and {(c['round'],c['fixture'],c['batch_size']) for c in data}==expected
for c in data:
 assert c['requests']==64 and len(c['call_seconds'])==64//c['batch_size']
 assert all(x>0 and math.isfinite(x) for x in c['call_seconds'])
 assert math.isclose(sum(c['call_seconds']),c['elapsed_seconds'],rel_tol=1e-9)
 assert math.isclose(64/c['elapsed_seconds'],c['requests_per_second'],rel_tol=1e-9)
 assert c['input_tokens']==sum(map(len,c['sample_prompt_token_ids']))*64//c['batch_size']
 assert math.isclose(c['input_tokens']/c['elapsed_seconds'],c['input_tokens_per_second'],rel_tol=1e-9)
 a=np.array(c['sample_vectors']);assert a.shape==(c['batch_size'],2048)
 assert np.isfinite(a).all() and np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5)
f=read(job/'summary.json');api=read(job/'api-summary.json')
assert api['passed'] and sum(c['count'] for c in api['cases'])==6 and api['invalid_role_status']==400
assert sum(c['count'] for c in f['comparisons'])==6
assert all(all(c['prompt_token_ids_equal']) for c in f['comparisons'])
assert min(v for c in f['comparisons'] for v in c['cosine_to_publisher_backend'])>0.99
old=read(R.parent/'Trident-Qwen3VL-2B/results/scnet/release-20261026T113612Z/native_reproduction/performance/summary.json')['cells']
old={(c['fixture'],c['batch_size']):c for c in old if c['round']==2};rows=[]
for c in data:
 if c['round']!=2:continue
 key=(c['fixture'],c['batch_size']);b=old[key]
 assert c['input_tokens']==b['input_tokens']
 first=next(x for x in data if (x['round'],x['fixture'],x['batch_size'])==(1,*key))
 rows.append({'fixture':key[0],'batch':key[1],'historical_rps':b['requests_per_second'],'optimized_rps':c['requests_per_second'],
  'gain_pct':100*(c['requests_per_second']/b['requests_per_second']-1),'input_tokens':c['input_tokens'],
  'input_tokens_per_second':c['input_tokens_per_second'],'mean_batch_call_ms':1000*np.mean(c['call_seconds']),
  'p99_batch_call_ms':1000*np.percentile(c['call_seconds'],99),
  'within_process_round_deviation_pct':100*(c['requests_per_second']/first['requests_per_second']-1)})
selection=read(job/'selection.json')
screen=R/'results'/selection['screen'];diagnostics=[]
baseline=read(screen/'baseline/performance/summary.json')['cells']
for candidate in selection['candidates']:
 name=candidate['name'];cells=read(screen/name/'performance/summary.json')['cells'];cos=[]
 for a,b in zip(baseline,cells):
  assert a['sample_prompt_token_ids']==b['sample_prompt_token_ids']
  cos.extend(np.sum(np.array(a['sample_vectors'])*np.array(b['sample_vectors']),axis=1).tolist())
 diagnostics.append({'candidate':name,'min_cosine_to_incidental_eager_screen':min(cos),'strict_0_9999_diagnostic':min(cos)>0.9999,'historical_0_99_functional_check':min(cos)>0.99})
report={'model':'Trident-Qwen3VL-2B-Optimized-v1','job':job.name,'selection':selection,
 'functional_inputs':6,'embedding_api_pass':True,'minimum_functional_cosine_to_publisher':min(v for c in f['comparisons'] for v in c['cosine_to_publisher_backend']),
 'measured_requests':sum(c['requests'] for c in data),'measured_input_tokens':sum(c['input_tokens'] for c in data),
 'cells':rows,'candidate_vector_diagnostics':diagnostics,
 'independent_process_reproduction':None,'official_accuracy_pass':None,
 'historical_accuracy':{'model_version':'original eager release','dataset':'ChartQA Origin','ndcg10_times100':69.4483,'rerun_for_optimized_version':False},
 'protocol':'One optimized process, two rounds, 12 cells each, 64 requests per cell, two warmup calls per cell. Second round reported. Embedding throughput, not a 608-request generation benchmark.'}
(R/'results/optimized-summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf8'))
print(json.dumps({k:v for k,v in report.items() if k not in ['selection','cells']},indent=2))
