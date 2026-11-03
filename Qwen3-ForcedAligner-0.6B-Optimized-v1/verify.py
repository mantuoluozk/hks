import json,math,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
O=R/'results'/Path(read(R/'results/current.json')['path']).name
assert read(O/'state.json')['status']=='completed'
assert read(O/'final.json')['exit']==0
assert read(O/'final/runtime/weights-check.json')['all_required_files_match']
s=read(O/'final/smoke/summary.json');assert s['functional_cases']==2
refs={x['id']:x for x in s['cases']}
entry=read(R/'results/entry-default/smoke/summary.json');assert entry['functional_cases']==2 and not entry['enforce_eager']
assert all(x['items']==refs[x['id']]['items'] for x in entry['cases'])
rows=[json.loads(x) for x in (O/'final/smoke/bench/requests.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==512 and sorted(x['request'] for x in rows)==list(range(512))
hist={k:read(R/'sources/baseline'/f'{k}.json') for k in refs}
max_hist=0
for x in rows:
 assert x['format_monotonicity_bounds_passed'] and x['batch_elapsed_seconds']>0
 assert len(x['raw_timestamp_bins'])==2*len(x['items']) and all(0<=v<5000 for v in x['raw_timestamp_bins'])
 assert x['classification_shape'][1]==5000
 times=[v for item in x['items'] for v in [item['start_time'],item['end_time']]]
 assert all(math.isfinite(v) for v in times) and all(0<=a<=b for a,b in zip(times,times[1:])) and max(times)<=x['audio_seconds']+0.16
 reference=refs[x['id']]['items'];delta=max(abs(a[k]-b[k]) for a,b in zip(reference,x['items']) for k in ['start_time','end_time'])
 assert math.isclose(delta,x['maximum_timestamp_delta_seconds'],abs_tol=1e-9)
 old=hist[x['id']];assert [z['text'] for z in old['items']]==[z['text'] for z in x['items']]
 max_hist=max(max_hist,max(abs(a[k]-b[k]) for a,b in zip(old['items'],x['items']) for k in ['start_time','end_time']))
workloads=read(O/'final/smoke/bench/summary.json')['workloads']
assert len(workloads)==8 and {(w['batch_size'],w['round']) for w in workloads}=={(b,r) for b in [1,2,4,8] for r in [1,2]}
for w in workloads:
 group=[x for x in rows if x['round']==w['round'] and x['batch_size']==w['batch_size']];assert len(group)==64
 wall=sum(group[i]['batch_elapsed_seconds'] for i in range(0,64,w['batch_size']))
 audio=sum(x['audio_seconds'] for x in group)
 assert math.isclose(wall,w['wall_seconds'],rel_tol=1e-9) and math.isclose(audio/wall,w['audio_seconds_per_wall_second'],rel_tol=1e-9)
for n in [1,2]:
 exported=[json.loads(x) for x in (R/'bench'/f'round{n}.jsonl').read_text(encoding='utf8').splitlines()]
 assert exported==[w for w in workloads if w['round']==n]
assert read(R/'env.log')['native_token_classification_import']
assert (R/'bench/requests.jsonl').read_bytes()==(O/'final/smoke/bench/requests.jsonl').read_bytes()
assert max_hist<=0.080001,('Timestamp difference exceeds historical 80ms observation',max_hist)
J=R/'results'/Path(read(R/'results/review-current.json')['path']).name
commands=read(J/'commands.json');assert len(commands)==6 and all(x['exit']==0 for x in commands)
assert read(J/'state.json')['all_exit_zero']
repeated=[json.loads(x) for x in (J/'bench/smoke/bench/requests.jsonl').read_text(encoding='utf8').splitlines()]
assert len(repeated)==512 and sorted(x['request'] for x in repeated)==list(range(512))
for w in read(J/'bench/smoke/bench/summary.json')['workloads']:
 group=[x for x in repeated if (x['batch_size'],x['round'])==(w['batch_size'],w['round'])];assert len(group)==64
 wall=sum(group[i]['batch_elapsed_seconds'] for i in range(0,64,w['batch_size']))
 assert math.isclose(sum(x['audio_seconds'] for x in group)/wall,w['audio_seconds_per_wall_second'],rel_tol=1e-9)
 assert all(x['format_monotonicity_bounds_passed'] and x['items']==refs[x['id']]['items'] for x in group)
subprocess.run([sys.executable,str(R/'scripts/verify_official_conditions.py'),'--job',str(J/'t03t04'),'--conditions',str(R/'sources/official-conditions.jsonl')],check=True)
result={'requests':512,'functional_cases':2,'review_requests':512,'t03t04_successful':30,'review_commands_exit_zero':6,'max_historical_boundary_difference_seconds':max_hist,'passed':True,'official_acceptance':None}
(R/'verify_result.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
