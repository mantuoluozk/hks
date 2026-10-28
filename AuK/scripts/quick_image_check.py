"""Check baked runtime and one real speech request, then stop the owned service."""
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import requests
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/new-image-quick-20261028';out.mkdir(parents=True,exist_ok=True)
record={'image_page':'https://www.scnet.cn/ui/aihub/image/zack233/2115345884192739329',
        'scope':'new Notebook image integrity + one real HTTP speech request',
        'full_performance_repeated':False,'official_accuracy_tested':False,'passed':False}
server=None
def save():
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
try:
    baked=Path('/opt/auk')
    manifest=json.loads((baked/'SHA256.json').read_text())
    for rel,digest in manifest.items():assert hashlib.sha256((baked/rel).read_bytes()).hexdigest()==digest,rel
    origin=Path(importlib.util.find_spec('sglang_omni').origin).resolve()
    assert origin.is_relative_to(baked),origin
    record.update(baked_files_verified=len(manifest),omni_origin=str(origin),
                  runtime_root=os.environ.get('HKS_AUK_RUNTIME'),
                  packages={p:importlib.metadata.version(p) for p in ['torch','sglang','transformers','x-transformers','openai']})
    record['phase']='server_startup';save()
    with (out/'server.log').open('wb') as log:
        server=subprocess.Popen(['bash',str(ROOT/'run.sh')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+360
        while True:
            if server.poll() is not None:raise RuntimeError('Owned image-check server exited')
            try:
                health=requests.get('http://127.0.0.1:30005/health',timeout=2)
                if health.ok and health.json().get('running'):break
            except requests.RequestException:pass
            if time.monotonic()>deadline:raise TimeoutError('Image-check startup timeout')
            time.sleep(2)
        body={'input':'你好，欢迎参加语音生成测试。','response_format':'wav','seed':42,
              'stage_params':{'auk_engine':{'gen_seconds':4}}}
        (out/'request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n')
        started=time.perf_counter()
        response=requests.post('http://127.0.0.1:30005/v1/audio/speech',json=body,timeout=180)
        record.update(http_status=response.status_code,request_seconds=time.perf_counter()-started)
        response.raise_for_status()
        (out/'chinese.wav').write_bytes(response.content)
        wav,rate=sf.read(io.BytesIO(response.content),dtype='float32')
        assert rate==24000 and len(wav)==96000 and np.isfinite(wav).all() and float(np.std(wav))>1e-6
        record.update(sample_rate=rate,samples=len(wav),waveform_sha256=hashlib.sha256(wav.tobytes()).hexdigest(),
                      wav_sha256=hashlib.sha256(response.content).hexdigest(),passed=True,phase='speech_passed')
except Exception as error:
    record.update(error=type(error).__name__+': '+str(error),phase='failed');raise
finally:
    if server is not None and server.poll() is None:
        server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.getpgid(server.pid)==server.pid:os.killpg(server.pid,signal.SIGKILL)
            server.wait(timeout=10)
    record['owned_server_stopped']=server is None or server.poll() is not None
    save()
