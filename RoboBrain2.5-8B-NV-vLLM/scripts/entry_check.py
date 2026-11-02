"""After the matrix has ended, check the shipped start/API/stop entrypoints."""
import json,os,re,subprocess,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
job=R/'results'/Path(json.loads((R/'results/formal-current.json').read_text(encoding='utf8'))['path']).name
assert json.loads((job/'state.json').read_text(encoding='utf8'))['status']=='completed','Do not interfere with a running matrix'
O=R/'results'/time.strftime('entry-%Y%m%dT%H%M%SZ',time.gmtime());O.mkdir()
env=dict(os.environ,RUN_RESULTS_DIR=str(O/'weights'));summary={'passed':False}
pidfile=R/'results/server.pid';before_pid=pidfile.read_text() if pidfile.exists() else None;started=False
try:
 with (O/'start.log').open('w') as f:subprocess.run(['bash','run.sh'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1900)
 started=True
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}));base='http://127.0.0.1:'+env['PORT']
 with op.open(base+'/health',timeout=10) as r:assert r.status==200
 with op.open(base+'/v1/models',timeout=10) as r:models=json.load(r)
 assert env['SERVED_NAME'] in [m['id'] for m in models['data']]
 body={'model':env['SERVED_NAME'],'messages':[{'role':'user','content':'What is 2 + 3? Answer with one number.'}],'max_tokens':8,'temperature':0}
 (O/'request.json').write_bytes(json.dumps(body,indent=2).encode())
 with op.open(urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=60) as r:response=json.load(r)
 (O/'response.json').write_bytes(json.dumps(response,indent=2).encode())
 assert re.fullmatch(r'\s*5[.!]?\s*',response['choices'][0]['message']['content'])
 summary={'passed':True,'health':200,'model':env['SERVED_NAME'],'answer':response['choices'][0]['message']['content'],'usage':response['usage']}
finally:
 after_pid=pidfile.read_text() if pidfile.exists() else None
 if started or after_pid!=before_pid:
  with (O/'stop.log').open('w') as f:stopped=subprocess.run(['bash','stop.sh'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
  summary['stop_exit']=stopped.returncode;summary['passed']=summary['passed'] and stopped.returncode==0
 else:summary['stop_exit']=None
 (O/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
 print(json.dumps({'path':str(O),**summary}),flush=True)
assert summary['passed']
