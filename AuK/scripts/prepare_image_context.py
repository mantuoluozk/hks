"""Build an offline SCNet context from verified sources and small wheels only."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'artifacts/image-context-v1'
if out.exists() and any(out.iterdir()):
    raise RuntimeError('Context already exists; use a new version instead of overwriting')
out.mkdir(parents=True, exist_ok=True)
payload = out/'auk'
payload.mkdir()

def copy(source, relative, digest=None):
    raw = source.read_bytes()
    if digest:
        assert hashlib.sha256(raw).hexdigest() == digest, source
    dest = payload/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)

original = json.loads((ROOT/'sources/sglang-omni-runtime-files.json').read_text())
compat = json.loads((ROOT/'sources/dtk-compat-files.json').read_text())
entries = {e['path']:e['sha256'] for e in original['files']}
entries.update({e['path']:e['sha256'] for e in compat['files']})
for rel,digest in entries.items():
    copy(ROOT/'runtime/sglang-omni'/rel, Path('sglang-omni')/rel, digest)
base = ROOT.parent/'Qwen-Image-2.1-Turbo'
base_wheels = json.loads((base/'sources/base-extra-wheel-files.json').read_text())
for entry in base_wheels:
    copy(ROOT.parents[1]/'../dockerFileTemp/qwen3-asr-1.7b/wheels'/entry['file'],
         Path('wheels')/entry['file'], entry['sha256'])
wheel = ROOT/'cache/wheels/x_transformers-1.31.14-py3-none-any.whl'
release = json.loads((ROOT/'sources/x-transformers-release.json').read_text())
published = next(item for item in release['urls'] if item['filename']==wheel.name)
copy(wheel, Path('wheels')/wheel.name, published['digests']['sha256'])
for filename in ['sglang-omni-runtime-files.json','dtk-compat-files.json']:
    copy(ROOT/'sources'/filename, Path('sources')/filename)
for source in [ROOT/'run.sh',ROOT/'run-scnet.sh',*sorted((ROOT/'scripts').glob('*.py'))]:
    copy(source, Path('model')/source.relative_to(ROOT))
hashes = {p.relative_to(payload).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in payload.rglob('*') if p.is_file()}
(payload/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
(payload/'verify_build.py').write_text('''import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/opt/auk')
for name,digest in json.loads((root/'SHA256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(root/'site-packages'),*[str(p) for p in sorted((root/'wheels').glob('*.whl'))]],check=True)
''')
docker = '''FROM image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18
COPY auk/ /opt/auk/
RUN python /opt/auk/verify_build.py
ENV HKS_AUK_RUNTIME=/opt/auk
'''
(out/'Dockerfile.auk').write_text(docker)
target = ROOT.parents[1].parent/'dockerFileTemp'
target.mkdir(exist_ok=True)
if (target/'auk').exists():
    raise RuntimeError('Platform auk context exists; inspect it before replacing')
shutil.copytree(payload,target/'auk')
(target/'Dockerfile.auk').write_text(docker)
for p in payload.rglob('*'):
    if p.is_file():
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((target/'auk'/p.relative_to(payload)).read_bytes()).digest()
result = {'dockerfile':'/public/home/zack233/dockerFileTemp/Dockerfile.auk',
          'files':len(hashes)+2,'bytes':sum(p.stat().st_size for p in payload.rglob('*') if p.is_file()),
          'payload_verified':True,'built':False,'weights_included':False}
(ROOT/'sources/image-context.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
