import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
O=R/'results'/time.strftime('command-review-%Y%m%dT%H%M%SZ',time.gmtime());O.mkdir()
(R/'results/review-current.json').write_text(json.dumps({'path':str(O)}))
rows=[]
commands=[('environment',['bash','env_check.sh']),('run',['bash','run.sh']),('smoke',['bash','smoke.sh']),('bench',['bash','bench.sh']),('t03t04',['bash','official.sh']),('verify',['python','verify.py'])]
for name,args in commands:
 env=dict(os.environ,RUN_RESULTS_DIR=str(O/name),OFFICIAL_OUTPUT_DIR=str(O/'t03t04'))
 (O/'state.json').write_text(json.dumps({'status':'running','stage':name}))
 with (O/(name+'.log')).open('w') as f:rc=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 rows.append({'stage':name,'command':args,'exit':rc});(O/'commands.json').write_text(json.dumps(rows,indent=2))
 if rc:raise SystemExit(rc)
(O/'state.json').write_text(json.dumps({'status':'completed','commands':len(rows),'all_exit_zero':True}))
