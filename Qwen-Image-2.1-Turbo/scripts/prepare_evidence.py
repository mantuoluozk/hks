"""Freeze already-completed local evidence; does not start a test or publish anything."""
import argparse
import hashlib
import json
from pathlib import Path
from image_gate import evaluate_image

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True)
for name in ['preflight','install-check','info']:p.add_argument('--image-'+name,type=Path)
a=p.parse_args()
validation=(ROOT/a.validation).resolve()
assert validation.is_relative_to((ROOT/'results').resolve())
status=json.loads((validation/'status.json').read_text())
assert status['status']=='completed','Do not freeze incomplete test results'
image=evaluate_image(ROOT,validation,vars(a))


def rel(path):return path.relative_to(ROOT.resolve()).as_posix()


paths=[]
for directory in ['scripts','sources','results']:
    for f in (ROOT/directory).rglob('*'):
        if (not f.is_file() or '__pycache__' in f.parts or f.suffix in {'.pyc','.pid'}
                or f.name in {'evidence-manifest.json','verify_result.json','paused-state.json'}):continue
        paths.append(f)
for name in ['README.md','runtime-config.json','Dockerfile.scnet','run-scnet.sh','run.sh','env_check.sh','smoke.sh','bench.sh','verify.py',
             '优化说明.md','测试说明.md','SCNet镜像重建.md']:
    paths.append(ROOT/name)
paths.extend(ROOT/image[k] for k in ['image_preflight','image_install_check','image_info'] if image.get(k))
manifest={'weights':'results/scnet/validation-20261026T123349Z/weights/weights-check.json',
          'configuration':'results/config-probe-fixed/config-comparison.json',
          'functional':rel(validation/'round-1/smoke/summary.json'),
          'service_validation':rel(validation/'status.json'),
          'request_traces':[rel(f) for f in sorted((validation/'round-1/trace').glob('requests-*.jsonl'))],
          'dedicated_image_revalidation':False,
          'sha256':{rel(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(paths))}}
manifest.update(image)
if image.get('image_info'):
    manifest['functional_rounds']=[rel(validation/f'round-{i}/smoke/summary.json') for i in [1,2]]
    manifest['request_trace_rounds']=[[rel(f) for f in sorted((validation/f'round-{i}/trace').glob('requests-*.jsonl'))] for i in [1,2]]
    assert all(manifest['request_trace_rounds'])
assert manifest['request_traces']
(ROOT/'results/evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(manifest['sha256']),'service_validation':manifest['service_validation']}))
