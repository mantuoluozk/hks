"""Prepare an offline build context using only the pinned runtime and model scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=ROOT/'artifacts/image-context')
p.add_argument('--base-extra',type=Path,default=Path('/root/private_data/dockerFileTemp/qwen3-asr-1.7b'))
args=p.parse_args();out=args.output.resolve()
assert out.is_relative_to((ROOT/'artifacts').resolve()),'Build context must stay inside this model artifacts directory'
out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):
    raise RuntimeError('Use an empty image-context directory to prevent stale build inputs')
manifest=json.loads((ROOT/'sources/runtime-dependencies.json').read_text())
for item in manifest['packages']:
    if not item['filename'].endswith('.whl'):continue
    src=ROOT/'runtime/wheels'/item['filename']
    assert hashlib.sha256(src.read_bytes()).hexdigest()==item['sha256']
    target=out/'runtime/wheels'/src.name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,target)
source=ROOT/'runtime/diffusers-py310'
for src in source.rglob('*'):
    if not src.is_file() or '__pycache__' in src.parts or src.suffix=='.pyc':continue
    target=out/'runtime/diffusers-py310'/src.relative_to(source)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
for name in ['run-scnet.sh','scripts/serve_sglang.py','scripts/preflight.py']:
    target=out/'model'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/name,target)
base_manifest=json.loads((ROOT/'sources/base-extra-wheel-files.json').read_text())
for item in base_manifest:
    src=args.base_extra/'wheels'/item['file']
    assert hashlib.sha256(src.read_bytes()).hexdigest()==item['sha256'],src
    target=out/'base-extra/wheels'/src.name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,target)
shutil.copyfile(ROOT/'sources/base-extra-requirements.txt',out/'base-extra/requirements.txt')
hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob('*')) if p.is_file()}
(out/'SHA256.json').write_text(json.dumps(hashes,indent=2))
(out/'verify_build.py').write_text('''import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/opt/qwen-image21')
manifest=json.loads((root/'SHA256.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
               '--target',str(root/'site-packages'),*[str(p) for p in sorted((root/'runtime/wheels').glob('*.whl'))]],check=True)
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
               '--find-links',str(root/'base-extra/wheels'),'--target','/opt/qwen3-asr-deps',
               '-r',str(root/'base-extra/requirements.txt')],check=True)
''')
(out/'Dockerfile').write_text('''FROM image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18
COPY qwen-image21-turbo/ /opt/qwen-image21/
RUN python /opt/qwen-image21/verify_build.py && mv /opt/qwen-image21/runtime/diffusers-py310 /opt/qwen-image21/diffusers-py310
ENV HKS_IMAGE_RUNTIME=/opt/qwen-image21
''')
(out/'.dockerignore').write_text('**/__pycache__/\n**/*.pyc\n')
print(json.dumps({'context':str(out),'files_verified':len(hashes),
                  'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                  'build_executed':False},indent=2))
