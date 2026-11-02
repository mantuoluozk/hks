"""Exercise the released run/preflight/one-request/stop path on the current image."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

import requests
from smoke_http import request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT / 'results' / time.strftime('entry-%Y%m%dT%H%M%SZ', time.gmtime())
OUT.mkdir()
env = dict(os.environ, HKS_PROFILE='optimized', RUN_RESULTS_DIR=str(OUT / 'preflight'))
(ROOT / 'results/entry-current.json').write_text(json.dumps({'path': str(OUT), 'controller_pid': os.getpid()}))
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
proc = None
report = {'profile': 'optimized', 'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
          for name in ['run.sh', 'run-scnet.sh', 'env.sh', 'stop.sh', 'serve-optimized.sh']},
          'official_acceptance': None}
try:
    with (OUT / 'server.log').open('w') as log:
        proc = subprocess.Popen(['bash', 'run.sh'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError('run.sh exited before readiness')
            try:
                with op.open('http://127.0.0.1:30004/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(5)
        else:
            raise TimeoutError('Entry readiness timeout')
        with requests.Session() as session:
            session.trust_env = False
            row = request(session, 'What is 2 + 2? Answer with the number only.', max_tokens=16)
        (OUT / 'arithmetic.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
        assert row['output'].strip() == '4' and row['finish_reason'] == 'stop'
        report['functional_passed'] = report['functional_total'] = 1
        report['weights_preflight'] = json.loads((OUT / 'preflight/summary.json').read_text())
        with (OUT / 'stop.log').open('w') as stoplog:
            subprocess.run(['bash', 'stop.sh'], stdout=stoplog, stderr=subprocess.STDOUT, timeout=90, check=True)
        proc.wait(timeout=30)
        report['owned_service_stopped'] = True
        report['passed'] = True
except Exception as exc:
    report.update(passed=False, error=repr(exc))
    raise
finally:
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=90)
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'path': str(OUT), 'passed': report.get('passed', False)}), flush=True)
