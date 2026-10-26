"""Validate one ASR model on the single SCNet card, keeping evidence separate."""
from datetime import datetime,timezone
import json,os,subprocess,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HKS=ROOT.parent.parent
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=ROOT/'results/scnet'/('validation-'+STAMP);OUT.mkdir(parents=True,exist_ok=False)
state={'created_at':STAMP,'execution_platform':'SCNet Notebook','artifacts':{},
       'official_accuracy_pass':None,'official_acceptance_passed':None}
env=dict(os.environ,RUN_RESULTS_DIR=str(OUT/'runtime'),SMOKE_OUTPUT_DIR=str(OUT/'smoke'),
         BENCH_OUTPUT_DIR=str(OUT/'bench'),EVAL_OUTPUT_DIR=str(OUT/'accuracy'),
         STREAMING_OUTPUT_DIR=str(OUT/'input-streaming'),ENFORCE_EAGER='0')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));server=None


def update(stage,**values):
    state.update(stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),**values)
    temp=OUT/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'status.json')
    print(json.dumps({'stage':stage,'evidence':str(OUT.relative_to(ROOT)),**values}),flush=True)


def run(label,args,*,cwd=ROOT,custom_env=None):
    with (OUT/(label+'.log')).open('wb') as f:
        subprocess.run(args,cwd=cwd,env=custom_env or env,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    global server
    update('preflight')
    run('weights',['bash','run-scnet.sh','python','scripts/check_weights.py'])
    run('dataset-preparation',['bash','run-scnet.sh','python','scripts/prepare_librispeech_parquet.py'])
    update('releasing_completed_mimo')
    mimo=ROOT.parent/'MiMo-V2.6-Distill-Qwen-9B'
    stop_env=dict(os.environ,MODEL_PATH=str(HKS/'weights/MiMo-V2.6-Distill-Qwen-9B'),
                  SERVED_NAME='mimo-v2.6-9b',PORT='30124',
                  RUN_RESULTS_DIR=str(mimo/'results/scnet/validation-20261026T021625Z/runtime'))
    old_pid=Path(stop_env['RUN_RESULTS_DIR'])/'server.pid'
    if old_pid.exists() and Path('/proc',old_pid.read_text().strip(),'cmdline').exists():
        run('stop-mimo',['bash','run-scnet.sh','bash','stop.sh'],cwd=mimo,custom_env=stop_env)
    else:
        (OUT/'stop-mimo.log').write_text('Completed MiMo service is already stopped; no signal sent.\n')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes().split(b'\0')
        except OSError:continue
        assert b'sglang.launch_server' not in cmd and b'vllm.entrypoints.openai.api_server' not in cmd, 'Another model service is active'
    run('environment',['bash','run-scnet.sh','python','scripts/inspect_environment.py'])
    update('starting_online_service')
    log=(OUT/'launcher.log').open('wb')
    server=subprocess.Popen(['bash','run-scnet.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,start_new_session=True)
    state['server_pid']=server.pid
    ready=False
    for _ in range(360):
        assert server.poll() is None,'ASR server exited during startup'
        try:
            with opener.open('http://127.0.0.1:30128/health',timeout=4) as r:assert r.status==200
            with opener.open('http://127.0.0.1:30128/v1/models',timeout=4) as r:models=json.load(r)
            ready=any(x['id']=='qwen3-asr-1.7b' for x in models['data'])
        except Exception:pass
        if ready:break
        time.sleep(5)
    assert ready,'ASR readiness timeout'
    update('online_smoke')
    run('smoke',['bash','run-scnet.sh','python','scripts/smoke_test.py'])
    smoke=json.loads((OUT/'smoke/summary.json').read_text())
    assert smoke['functional_cases']==6 and all(smoke['cross_interface_exact'].values())
    state['artifacts']['smoke']=str((OUT/'smoke').relative_to(ROOT))
    update('audio_performance')
    run('audio-bench',['bash','run-scnet.sh','python','scripts/bench_audio.py'])
    bench=json.loads((OUT/'bench/summary.json').read_text())
    assert len(bench['workloads'])==8 and bench['counted_requests']==128
    state['artifacts']['bench']=str((OUT/'bench').relative_to(ROOT))
    update('public_test_clean_wer')
    run('public-wer',['bash','run-scnet.sh','python','scripts/evaluate_librispeech.py'])
    accuracy=json.loads((OUT/'accuracy/summary.json').read_text())
    assert accuracy['complete'] and accuracy['completed']==accuracy['requested']==2620 and accuracy['truncated']==0
    state['artifacts']['accuracy']=str((OUT/'accuracy').relative_to(ROOT))
    update('offline_pcm_input_streaming')
    run('stop-online',['bash','run-scnet.sh','bash','stop.sh']);server.wait(timeout=90);server=None
    run('input-streaming',['bash','run-scnet.sh','python','scripts/streaming_offline.py'])
    streaming=json.loads((OUT/'input-streaming/summary.json').read_text());assert streaming['completed']==4
    state['artifacts']['input_streaming']=str((OUT/'input-streaming').relative_to(ROOT))
    update('platform_tests_complete',public_wer_percent=accuracy['wer_percent'],
           note='Image reconstruction and report/publication are tracked separately')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        update('failed',error=type(e).__name__+': '+str(e))
        if server is not None and server.poll() is None:
            try:run('stop-failed-server',['bash','run-scnet.sh','bash','stop.sh'])
            except Exception as x:print('Cleanup:',repr(x),flush=True)
        raise
    finally:run('backup-shared',['python',str(HKS/'platform/scnet-notebook/backup_shared.py')])
