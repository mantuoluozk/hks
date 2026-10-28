"""Prepare a standalone offline context from pre-fetched pinned source and wheels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--source-dir', type=Path, required=True, help='clean pinned SGLang-Omni checkout')
p.add_argument('--wheel-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True, help='new, empty build directory')
a = p.parse_args()
if a.output.exists() and any(a.output.iterdir()):
    raise RuntimeError('Use a new empty output directory')
runtime = a.output/'auk/sglang-omni'
runtime.mkdir(parents=True, exist_ok=True)
for record in json.loads((ROOT/'sources/sglang-omni-runtime-files.json').read_text())['files']:
    raw = (a.source_dir/record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']
    dest = runtime/record['path']; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
# Reuse the audited patch with its normal model-local directory convention.
patch_root = a.output/'patch-work'
(patch_root/'runtime').mkdir(parents=True)
shutil.copytree(ROOT/'sources', patch_root/'sources')
shutil.copytree(runtime, patch_root/'runtime/sglang-omni')
(patch_root/'scripts').mkdir()
shutil.copyfile(ROOT/'scripts/apply_dtk_compat.py', patch_root/'scripts/apply_dtk_compat.py')
subprocess.run([sys.executable, str(patch_root/'scripts/apply_dtk_compat.py')], check=True)
for record in json.loads((ROOT/'sources/dtk-compat-files.json').read_text())['files']:
    raw = (patch_root/'runtime/sglang-omni'/record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    dest = runtime/record['path']; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
assert patch_root.resolve().is_relative_to(a.output.resolve())
shutil.rmtree(patch_root)
payload = a.output/'auk'
for record in json.loads((ROOT/'sources/runtime-wheels.json').read_text()):
    raw = (a.wheel_dir/record['file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['file']
    dest = payload/'wheels'/record['file']; dest.parent.mkdir(exist_ok=True); dest.write_bytes(raw)
for name in ['run.sh','run-scnet.sh']:
    dest = payload/'model'/name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(ROOT/name,dest)
shutil.copytree(ROOT/'scripts', payload/'model/scripts', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(ROOT/'sources', payload/'sources')
hashes = {x.relative_to(payload).as_posix():hashlib.sha256(x.read_bytes()).hexdigest()
          for x in payload.rglob('*') if x.is_file()}
(payload/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
shutil.copyfile(ROOT/'sources/image-build/auk/verify_build.py',payload/'verify_build.py')
shutil.copyfile(ROOT/'sources/image-build/Dockerfile.auk',a.output/'Dockerfile.auk')
print(json.dumps({'output':str(a.output),'files':len(hashes)+2,'weights_included':False,
                  'rebuilt_image_tested':False}))
