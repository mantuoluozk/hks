"""Create a small offline SCNet image context; excludes weights and evidence."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=Path('/public/home/zack233/dockerFileTemp')
out=base/'recon2reason-vllm'
out.mkdir(parents=True,exist_ok=False)
shutil.copytree('/opt/qwen3-asr-deps',out/'deps',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(out/'model/scripts').mkdir(parents=True)
shutil.copy2(root/'bootstrap.sh',out/'model/bootstrap.sh')
shutil.copy2(root/'scripts/smoke_test.py',out/'model/scripts/smoke_test.py')
(out/'platform').mkdir()
shutil.copy2('/root/private_data/hks/platform/scnet-notebook/env-scnet.sh',out/'platform/env-scnet.sh')
verify='''import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
rows=json.loads((r/'manifest.json').read_text())
for x in rows:
 p=r/x['path']
 assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x
print('Verified',len(rows),'build files')
'''
(out/'verify.py').write_text(verify)
rows=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file()]
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
docker='''FROM image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18
COPY recon2reason-vllm/ /opt/recon2reason-vllm/
RUN python /opt/recon2reason-vllm/verify.py && cp -a /opt/recon2reason-vllm/deps /opt/qwen3-asr-deps
ENV PYTHONPATH=/opt/qwen3-asr-deps
# Keep the base image's Notebook/SSH startup. Model weights stay on shared storage.
'''
target=base/'Dockerfile.recon2reason-vllm'
with target.open('x') as f:f.write(docker)
(root/'Dockerfile.scnet').write_text(docker)
(root/'results/image-context.json').write_text(json.dumps(dict(dockerfile=str(target),context=str(base),files=len(rows),bytes=sum(r['bytes'] for r in rows),manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()),indent=2))
exec(compile(verify,str(out/'verify.py'),'exec'),{'__file__':str(out/'verify.py')})
print(target)
