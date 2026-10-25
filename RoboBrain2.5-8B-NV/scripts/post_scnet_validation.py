"""Finish diagnostics and prepare reviewable materials after the running matrix."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage(name, **extra):
    value = {'stage': name, 'updated_at': datetime.now(timezone.utc).isoformat(),
             'execution_platform': 'SCNet Notebook', 'official_acceptance_passed': None, **extra}
    (ROOT / 'results/scnet/finalization-status.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controller-pid', type=int)
    ap.add_argument('--wait-seconds', type=int, default=14400)
    args = ap.parse_args()
    os.chdir(ROOT)
    assert os.environ.get('EXECUTION_PLATFORM') == 'SCNet Notebook'
    runtime = json.loads((ROOT / 'results/scnet/runtime/runtime-config.json').read_text())
    assert runtime['HIP_VISIBLE_DEVICES'] == '0'
    stage('waiting-for-matrix')
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            folder = ROOT / (ROOT / 'results/scnet/benchmark-path.txt').read_text().strip()
            audit = json.loads((folder / 'audit.json').read_text())
            if audit.get('complete') and len(audit.get('workloads', [])) == 12:
                break
        except (OSError, ValueError):
            pass
        if args.controller_pid and not Path(f'/proc/{args.controller_pid}').exists():
            raise RuntimeError('Matrix controller exited before a complete audit; inspect benchmark.log')
        if time.monotonic() >= deadline:
            raise TimeoutError('Matrix did not complete within the wait budget')
        time.sleep(15)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    smoke = sorted((ROOT / 'results/scnet').glob('smoke-*/summary.json'))[-1].parent
    env = dict(os.environ, REFERENCE_SMOKE_DIR=str(smoke),
               DIAGNOSTIC_OUTPUT_DIR=str(ROOT / 'results/scnet' / ('stability-' + stamp)),
               CONCURRENCY_OUTPUT_DIR=str(ROOT / 'results/scnet' / ('concurrency-' + stamp)))
    stage('same-service-diagnostics')
    with (ROOT / 'results/scnet/stability.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/cache_diagnostic.py'], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    with (ROOT / 'results/scnet/concurrency.log').open('w') as log:
        concurrent = subprocess.run([sys.executable, 'scripts/concurrency_diagnostic.py'], env=env, stdout=log, stderr=subprocess.STDOUT)
    # Consistency is reported separately; it is never relabelled as GPU accuracy.
    stage('generating-report', same_service_concurrency_exit=concurrent.returncode)
    subprocess.run([sys.executable, 'scripts/finalize_scnet_results.py'], check=True)
    stage('report-ready-for-review', same_service_concurrency_exit=concurrent.returncode,
          github_published=False, sourcefind_published=False)
    subprocess.run([sys.executable, 'scripts/package_scnet_evidence.py'], check=True)
    subprocess.run([sys.executable, os.environ['HKS_ROOT'] + '/platform/scnet-notebook/backup_shared.py'], check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        stage('failed', error_type=type(exc).__name__, error=str(exc))
        raise
