"""Check the dedicated image payload and base runtime, without loading weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--image', required=True)
parser.add_argument('--image-page', required=True)
args = parser.parse_args()
payload = Path('/opt/kimi-vl-vllm')
manifest = payload / 'manifest.json'
rows = json.loads(manifest.read_text())
checks = 0
for row in rows:
    paths = [payload / row['path']]
    if row['path'].startswith('deps/'):
        paths.append(Path('/opt/qwen3-asr-deps') / row['path'][5:])
    for path in paths:
        assert path.stat().st_size == row['bytes'], str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
        checks += 1
expected = json.loads((ROOT / 'results/runtime-config.json').read_text())
packages = {k: importlib.metadata.version(k) for k in expected['packages']}
assert packages == expected['packages'], packages
for path, sha in expected['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
report = {'image': args.image, 'image_page': args.image_page, 'payload_files': len(rows),
          'file_checks': checks, 'packages': packages, 'framework_sources_match': True,
          'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
          'verified': True, 'model_inference_verified': False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
