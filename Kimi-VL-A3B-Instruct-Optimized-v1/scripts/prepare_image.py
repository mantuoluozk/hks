"""Prepare offline model code/API dependencies for the SCNet image builder."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/public/home/zack233/dockerFileTemp')
OUT = BASE / 'kimi-vl-vllm'
OUT.mkdir(parents=True, exist_ok=False)
shutil.copytree('/opt/qwen3-asr-deps', OUT / 'deps', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
(OUT / 'model').mkdir()
for folder in ['scripts', 'sources']:
    shutil.copytree(ROOT / folder, OUT / 'model' / folder,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for file in sorted(ROOT.glob('*.sh')):
    shutil.copy2(file, OUT / 'model' / file.name)
for file in ['verify.py', 'verify_formal.py']:
    shutil.copy2(ROOT / file, OUT / 'model' / file)
(OUT / 'platform').mkdir()
shutil.copy2('/root/private_data/hks/platform/scnet-notebook/env-scnet.sh', OUT / 'platform/env-scnet.sh')
verify = '''import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
rows=json.loads((r/'manifest.json').read_text())
for x in rows:
 p=r/x['path']
 assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
print('Verified',len(rows),'build files')
'''
(OUT / 'verify_build.py').write_text(verify)
rows = [{'path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size,
         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(OUT.rglob('*')) if p.is_file()]
(OUT / 'manifest.json').write_text(json.dumps(rows, indent=2))
docker = '''FROM image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18
COPY kimi-vl-vllm/ /opt/kimi-vl-vllm/
RUN python /opt/kimi-vl-vllm/verify_build.py && cp -a /opt/kimi-vl-vllm/deps /opt/qwen3-asr-deps
ENV PYTHONPATH=/opt/qwen3-asr-deps
# Inherit Notebook/SSH startup; load weights and data from shared storage.
'''
target = BASE / 'Dockerfile.kimi-vl-vllm'
with target.open('x') as f:
    f.write(docker)
(ROOT / 'Dockerfile.scnet').write_text(docker)
subprocess.run(['python', str(OUT / 'verify_build.py')], check=True)
record = {'dockerfile': str(target), 'context': str(BASE), 'payload_files': len(rows),
          'bytes': sum(r['bytes'] for r in rows), 'verified': True,
          'manifest_sha256': hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest(),
          'image_built': False, 'fresh_notebook_validated': False}
(ROOT / 'results/image-context-20261102.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
