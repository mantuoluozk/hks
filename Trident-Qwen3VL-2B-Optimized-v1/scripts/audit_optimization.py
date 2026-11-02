"""Recompute embedding throughput and compare actual inputs/vectors across candidates."""
import argparse,json,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('job',type=Path);a=p.parse_args()
def read(path):
 data=json.loads(path.read_text());out={}
 for r in data['cells']:
  key=(r['round'],r['fixture'],r['batch_size']);assert key not in out
  assert r['requests']==64 and len(r['call_seconds'])==64//r['batch_size']
  assert all(x>0 for x in r['call_seconds'])
  assert math.isclose(sum(r['call_seconds']),r['elapsed_seconds'],rel_tol=1e-9)
  assert math.isclose(64/r['elapsed_seconds'],r['requests_per_second'],rel_tol=1e-9)
  assert r['input_tokens']==sum(map(len,r['sample_prompt_token_ids']))*(64//r['batch_size'])
  v=np.array(r['sample_vectors']);assert v.shape==(r['batch_size'],2048)
  assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
  out[key]=r
 return out
baseline=a.job/'baseline/performance/summary.json'
if not baseline.exists():baseline=Path(__file__).resolve().parents[1]/'results/optimization-20261102T080346Z/baseline/performance/summary.json'
b=read(baseline);assert len(b)==12
reports={}
for name in ['inprocess','piecewise-inprocess','inprocess-cpu1','inprocess-cpu2','piecewise-best-cpu']:
 q=a.job/name/'performance/summary.json'
 if not q.exists():continue
 c=read(q);assert set(b)==set(c);rows=[]
 for k,x in b.items():
  y=c[k];assert x['sample_prompt_token_ids']==y['sample_prompt_token_ids']
  va=np.array(x['sample_vectors']);vb=np.array(y['sample_vectors'])
  cosine=np.sum(va*vb,axis=1);assert np.min(cosine)>0.99  # Historical functional criterion; strict diagnostic retained below
  rows.append({'fixture':k[1],'batch':k[2],'baseline_rps':x['requests_per_second'],'candidate_rps':y['requests_per_second'],
    'gain_pct':100*(y['requests_per_second']/x['requests_per_second']-1),'min_cosine':float(np.min(cosine)),'strict_0_9999_diagnostic':bool(np.min(cosine)>0.9999),'historical_0_99_functional_check':True,
    'max_absolute_diff':float(np.max(np.abs(va-vb))),'input_tokens_equal':True})
 reports[name]=rows
(a.job/'comparison.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
