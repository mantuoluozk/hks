import json,os,subprocess,time,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
O=R/'results'/time.strftime('optimization-%Y%m%dT%H%M%SZ',time.gmtime());O.mkdir(parents=True)
(R/'results/current.json').write_text(json.dumps({'path':str(O)}))
def state(**kw):(O/'state.json').write_text(json.dumps(kw,indent=2))
def run(name,settings,count,rounds):
 out=O/name;env=dict(os.environ,**settings,RUN_RESULTS_DIR=str(out),BATCH_BENCH='1',ALIGN_COUNT=str(count),ALIGN_ROUNDS=str(rounds))
 state(status='running',stage=name)
 with (O/(name+'.log')).open('w') as f:
  rc=subprocess.run(['bash','run-scnet.sh','bash','run.sh'],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 result={'name':name,'settings':settings,'exit':rc}
 if rc==0:
  b=json.loads((out/'smoke/bench/summary.json').read_text());result['workloads']=b['workloads'];result['score']=sum(x['audio_seconds_per_wall_second'] for x in b['workloads'])/len(b['workloads'])
 return result
candidates=[('mp0-eager4',{'VLLM_ENABLE_V1_MULTIPROCESSING':'0','ALIGN_EAGER':'1','OMP_NUM_THREADS':'4'}),('mp0-piecewise4',{'VLLM_ENABLE_V1_MULTIPROCESSING':'0','ALIGN_EAGER':'0','OMP_NUM_THREADS':'4','ALIGN_COMPILATION':'{"mode":3,"cudagraph_mode":"PIECEWISE"}' }),('mp0-piecewise2',{'VLLM_ENABLE_V1_MULTIPROCESSING':'0','ALIGN_EAGER':'0','OMP_NUM_THREADS':'2','ALIGN_COMPILATION':'{"mode":3,"cudagraph_mode":"PIECEWISE"}'})]
previous=R/'results/optimization-20261103T013804Z'
shutil.copytree(previous/'mp0-eager4',O/'mp0-eager4')
results=json.loads((previous/'screen.json').read_text())[:1]
for name,settings in candidates[1:]:
 results.append(run(name,settings,8,1));(O/'screen.json').write_text(json.dumps(results,indent=2))
valid=[x for x in results if x['exit']==0 and max(w['maximum_timestamp_delta_seconds'] for w in x['workloads'])<=0.080001]
assert valid,'No valid optimization candidate'
best=max(valid,key=lambda x:x['score']);(O/'selected.json').write_text(json.dumps(best,indent=2))
final=run('final',best['settings'],64,2);(O/'final.json').write_text(json.dumps(final,indent=2));assert final['exit']==0
state(status='completed',selected=best['name'])
