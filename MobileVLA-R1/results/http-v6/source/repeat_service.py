"""Repeat the same functional prefix and workload in a second service process."""
import argparse,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--first',default='http-v1');p.add_argument('--second',default='http-v2');args=p.parse_args()
    first=ROOT/'results'/args.first;second=ROOT/'results'/args.second;second.mkdir(parents=True,exist_ok=False)
    assert json.loads((first/'checks/summary.json').read_text())['passed']
    json.loads((first/'performance/summary.json').read_text())
    pid=int((first/'server.pid').read_text())
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'scripts/serve.py' in command
    assert Path(f'/proc/{pid}/cwd').resolve()==ROOT
    os.kill(pid,signal.SIGTERM)
    for _ in range(60):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(1)
    else:raise RuntimeError('First service did not exit')
    log=(second/'server.log').open('w')
    service=subprocess.Popen(['bash','run-scnet.sh','python','scripts/serve.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (second/'server.pid').write_text(str(service.pid)+'\n')
    try:
        for _ in range(180):
            assert service.poll() is None,'Server exited before readiness'
            try:
                with urllib.request.urlopen('http://127.0.0.1:30006/health',timeout=2) as r:
                    if json.load(r)['ready']:break
            except OSError:pass
            time.sleep(1)
        else:raise RuntimeError('Readiness timed out')
        for script,dest in [('check_http.py','checks'),('bench_http.py','performance')]:
            subprocess.run(['bash','run-scnet.sh','python','scripts/'+script,'--out',str(second/dest)],cwd=ROOT,check=True,timeout=600)
        old=json.loads((first/'performance/summary.json').read_text())['results']
        new=json.loads((second/'performance/summary.json').read_text())['results']
        rows=[]
        for a,b in zip(old,new):
            assert a['images']==b['images']
            delta=abs(b['mean_seconds']-a['mean_seconds'])/a['mean_seconds']*100
            rows.append({'images':a['images'],'first_mean_seconds':a['mean_seconds'],
                         'second_mean_seconds':b['mean_seconds'],'deviation_percent':delta})
        (ROOT/'results/http-reproducibility.json').write_text(json.dumps({'independent_processes':True,
            'same_functional_prefix':True,'first':args.first,'second':args.second,'results':rows,'within_20_percent':all(x['deviation_percent']<=20 for x in rows),
            'official_accuracy_tested':False},indent=2)+'\n')
    finally:
        service.terminate();service.wait(timeout=60);log.close()
        with (second/'backup.log').open('w') as backup:
            subprocess.run(['python',str(ROOT.parents[1]/'platform/scnet-notebook/backup_shared.py')],cwd=ROOT,stdout=backup,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
