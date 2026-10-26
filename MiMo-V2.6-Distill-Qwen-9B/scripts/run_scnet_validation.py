"""Serial SCNet validation; wait for the existing RoboBrain evaluation first.

This worker never changes weights or declares official acceptance. All outputs
stay under this model's shared-storage directory. No Docker is launched.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HKS = ROOT.parent.parent
ROBO = ROOT.parent/'RoboBrain2.5-8B-NV'
ROBO_RESULT = ROBO/'results/scnet/publisher-cvbench-full-20261026/summary.json'
ROBO_RUNTIME = ROBO/'results/scnet/publisher-runtime-c64-20261026'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = ROOT/'results/scnet'/('validation-'+STAMP)
OUT.mkdir(parents=True, exist_ok=False)
STATUS = OUT/'status.json'
state = {'created_at':STAMP, 'execution_platform':'SCNet Notebook', 'official_accuracy_pass':None,
         'official_acceptance_passed':None, 'artifacts':{}, 'stage':'created'}
env = dict(os.environ, RUN_RESULTS_DIR=str(OUT/'runtime'), SMOKE_OUTPUT_DIR=str(OUT/'smoke'))
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
launcher = None
parser=argparse.ArgumentParser()
parser.add_argument('--wait-robobrain-pid',type=int)
args=parser.parse_args()


def update(stage, **values):
    state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATUS)
    print(json.dumps({'stage':stage, **values}), flush=True)


def run(label, command, *, cwd=ROOT, runtime_env=None):
    with (OUT/(label+'.log')).open('wb') as f:
        subprocess.run(command,cwd=cwd,env=runtime_env or env,stdout=f,stderr=subprocess.STDOUT,check=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global launcher
    update('waiting_for_robobrain', prerequisite=str(ROBO_RESULT.relative_to(HKS)))
    while args.wait_robobrain_pid is not None:
        if ROBO_RESULT.exists():
            summary=json.loads(ROBO_RESULT.read_text())
            assert summary['full_test'] and summary['completed']==summary['expected']==2638
            assert summary['errors']==0, 'RoboBrain finished with errors; inspect before switching models'
            # Its shell also performs an end-of-run backup. Wait for that shell to exit.
            p=Path(f'/proc/{args.wait_robobrain_pid}/cmdline')
            if not p.exists() or b'run_public_cvbench.sh' not in p.read_bytes():
                state['robobrain_summary']=summary
                break
        else:
            p=Path(f'/proc/{args.wait_robobrain_pid}/cmdline')
            assert p.exists() and b'run_public_cvbench.sh' in p.read_bytes(), 'RoboBrain worker ended without full results'
        time.sleep(10)
    if args.wait_robobrain_pid is not None:
        update('releasing_robobrain')
        robo_env=dict(os.environ,RUN_RESULTS_DIR=str(ROBO_RUNTIME))
        run('stop-robobrain',['bash','run-scnet.sh','bash','stop.sh'],cwd=ROBO,runtime_env=robo_env)
    # Do not alter shared framework files while any model service is alive.
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: cmd=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        assert b'sglang.launch_server' not in cmd, f'Another SGLang service is alive: {p.parent.name}'
    framework=Path('/usr/local/lib/python3.10/dist-packages/sglang/srt/batch_invariant_ops/batch_invariant_ops.py')
    original=ROBO/'cache/deterministic-patch-original.py'
    base_hash='9a9fdb89d84f8c392ae9a46f46c0a846b332e78f4ae77c24f240f8db55e6a6e4'
    current=sha(framework)
    assert current in {base_hash,'371f9816ae1614d944dce1616ee2e88a6d1b6dfed2ca660275ae28cb9785bb39'}
    if current!=base_hash:
        assert sha(original)==base_hash
        framework.write_bytes(original.read_bytes())
    assert sha(framework)==base_hash
    (OUT/'framework-source-check.json').write_text(json.dumps({
        'source_before_sha256':current,'source_after_sha256':base_hash,'matches_base_image':True,
        'deterministic_inference_enabled_for_mimo':False},indent=2))
    update('preflight')
    run('data-verification',['bash','run-scnet.sh','python','scripts/prepare_scnet_eval.py'])
    run('environment-capture',['bash','run-scnet.sh','bash','env_check.sh'])
    run('device-check',['bash','run-scnet.sh','python','-c',
        'import torch,json; assert torch.cuda.device_count()==1; assert torch.cuda.mem_get_info()[0]>48*1024**3,"GPU is not idle"; print(json.dumps({"devices":torch.cuda.device_count(),"name":torch.cuda.get_device_name(0),"free_total_bytes":torch.cuda.mem_get_info()}))'])
    update('starting_service')
    log=(OUT/'launcher.log').open('wb')
    launcher=subprocess.Popen(['bash','run-scnet.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,start_new_session=True)
    state['launcher_pid']=launcher.pid;state['artifacts']['runtime']=str((OUT/'runtime').relative_to(ROOT))
    ready=False
    for _ in range(360):
        assert launcher.poll() is None, 'MiMo launcher exited; inspect launcher and server logs'
        try:
            with opener.open('http://127.0.0.1:30124/health',timeout=4) as response:
                assert response.status == 200
            with opener.open('http://127.0.0.1:30124/v1/models',timeout=3) as response:
                models=json.load(response)
            ready=any(x['id']=='mimo-v2.6-9b' for x in models['data'])
        except Exception: pass
        if ready: break
        time.sleep(5)
    assert ready, 'MiMo readiness timeout'
    update('smoke')
    run('smoke',['bash','run-scnet.sh','python','scripts/smoke.py'])
    smoke=json.loads((OUT/'smoke/summary.json').read_text())
    assert smoke['passed']==smoke['total']==11
    state['artifacts']['smoke']=str((OUT/'smoke').relative_to(ROOT))
    for profile in ('smoke','matrix'):
        update('benchmark_'+profile)
        before=set((ROOT/'bench').glob(profile+'-*'))
        run('benchmark-'+profile,['bash','run-scnet.sh','bash','bench.sh'],runtime_env=dict(env,BENCH_PROFILE=profile))
        added=set((ROOT/'bench').glob(profile+'-*'))-before
        assert len(added)==1, 'Ambiguous benchmark directory'
        folder=added.pop();audit=json.loads((folder/'audit.json').read_text())
        assert audit['complete'] and audit['cases']==(4 if profile=='smoke' else 12)
        state['artifacts']['benchmark_'+profile]=str(folder.relative_to(ROOT))
    update('public_quality')
    run('public-quality',['bash','run-scnet.sh','python','scripts/public_eval.py',
                          '--output',str(OUT/'quality'),'--workers','8'])
    quality=json.loads((OUT/'quality/summary.json').read_text())
    assert all(x['total']==200 and x['errors']==0 for x in quality['datasets'].values())
    state['artifacts']['quality']=str((OUT/'quality').relative_to(ROOT))
    state['artifacts']['validation_summary']=str((OUT/'validation-summary.json').relative_to(ROOT))
    state['functional']=smoke;state['public_quality']=quality
    update('platform_tests_complete', note='Report, image reproduction and repository publication are tracked separately')
    run('summarize-results',['python','scripts/summarize_scnet.py',str(OUT)])


if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        update('failed',error=type(e).__name__+': '+str(e))
        if launcher is not None and launcher.poll() is None:
            try:run('stop-failed-mimo',['bash','run-scnet.sh','bash','stop.sh'])
            except Exception as cleanup_error:print('Cleanup:',repr(cleanup_error),flush=True)
        raise
    finally:
        run('backup-shared',['python',str(HKS/'platform/scnet-notebook/backup_shared.py')])
