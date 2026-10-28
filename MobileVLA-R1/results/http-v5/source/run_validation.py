"""Run the official RGB helper reference and two isolated HTTP service processes."""
import hashlib,json,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def command(arguments,log,timeout=600):
    with log.open('w') as stream:
        subprocess.run(['bash','run-scnet.sh','python',*arguments],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def main():
    state=ROOT/'results/validation-current.json'
    def status(stage):state.write_text(json.dumps({'stage':stage,'time_unix':time.time()},indent=2)+'\n')
    try:
        status('reference-v5')
        command(['scripts/probe_reference.py'],ROOT/'results/reference-v5.log')
        for name in ['http-v5','http-v6']:
            base=ROOT/'results'/name;base.mkdir(parents=True,exist_ok=False)
            snapshot=base/'source';snapshot.mkdir()
            manifest=[]
            for path in sorted((ROOT/'scripts').glob('*.py')):
                raw=path.read_bytes();(snapshot/path.name).write_bytes(raw)
                manifest.append({'path':path.name,'sha256':hashlib.sha256(raw).hexdigest()})
            (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            status(name+'-starting')
            stream=(base/'server.log').open('w')
            service=subprocess.Popen(['bash','run-scnet.sh','python','scripts/serve.py'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            (base/'server.pid').write_text(str(service.pid)+'\n')
            try:
                for _ in range(180):
                    assert service.poll() is None,'Service failed before readiness'
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:30006/health',timeout=2) as r:
                            if json.load(r)['ready']:break
                    except OSError:pass
                    time.sleep(1)
                else:raise RuntimeError('Readiness timeout')
                status(name+'-checking')
                command(['scripts/check_http.py','--out',str(base/'checks')],base/'checks.log')
                command(['scripts/bench_http.py','--out',str(base/'performance')],base/'performance.log')
            finally:
                if service.poll() is None:service.terminate()
                service.wait(timeout=60);stream.close()
        values=[json.loads((ROOT/'results'/name/'performance/summary.json').read_text())['results'] for name in ['http-v5','http-v6']]
        rows=[]
        for old,new in zip(*values):
            assert old['images']==new['images']
            rows.append({'images':old['images'],'first_mean_seconds':old['mean_seconds'],
                'second_mean_seconds':new['mean_seconds'],
                'deviation_percent':abs(new['mean_seconds']-old['mean_seconds'])/old['mean_seconds']*100})
        (ROOT/'results/http-reproducibility.json').write_text(json.dumps({'independent_processes':True,
            'same_functional_prefix':True,'first':'http-v5','second':'http-v6','results':rows,
            'within_20_percent':all(x['deviation_percent']<=20 for x in rows),'official_accuracy_tested':False},indent=2)+'\n')
        command(['scripts/verify.py'],ROOT/'results/selfcheck.log')
        status('completed')
    except Exception as e:
        status('failed: '+repr(e));raise
    finally:
        with (ROOT/'results/validation-backup.log').open('w') as stream:
            subprocess.run(['python',str(ROOT.parents[1]/'platform/scnet-notebook/backup_shared.py')],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
