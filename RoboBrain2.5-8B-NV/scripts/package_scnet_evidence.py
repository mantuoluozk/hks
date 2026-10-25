"""Freeze SCNet evidence only; keep original logs and redact private IPs in exports."""
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = set()
files.update(p for p in (root / 'results/scnet').rglob('*')
             if p.is_file() and not p.is_symlink() and p.suffix != '.pid'
             and p.name != 'benchmark-path.txt'
             and p.name != 'evidence-manifest.json')
for p in (root / 'bench').glob('scnet-matrix-*/config.txt'):
    if 'source=SCNet Notebook' in p.read_text():
        files.update(f for f in p.parent.iterdir() if f.is_file() and not f.is_symlink())
runtime = json.loads((root / 'results/scnet/runtime/runtime-config.json').read_text())
assert runtime['CARD'] == 'BW' and runtime['HIP_VISIBLE_DEVICES'] == '0'
assert runtime['EXECUTION_PLATFORM'] == 'SCNet Notebook'
files.add(root / 'env.log')
pattern = re.compile(rb'(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9])')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
manifest, content = [], {}
for p in sorted(files):
    rel = p.relative_to(root).as_posix()
    assert not any(x in {'cache', 'data', 'weights'} for x in p.relative_to(root).parts)
    original = p.read_bytes()
    exported, count = pattern.subn(b'<internal-ip>', original) if p.suffix == '.log' else (original, 0)
    content[rel] = exported
    manifest.append({'path': rel, 'bytes': len(exported), 'sha256': hashlib.sha256(exported).hexdigest(),
                     'original_sha256': hashlib.sha256(original).hexdigest(), 'private_ip_redactions': count})
payload = {'created_at': stamp, 'execution_platform': 'SCNet Notebook', 'files': manifest,
           'official_acceptance_passed': None,
           'policy': 'Original shared-storage files remain unchanged. Only RFC1918 IPs in exported logs are redacted. Weights, datasets and credentials excluded.'}
content['results/scnet/evidence-manifest.json'] = (json.dumps(payload, indent=2) + '\n').encode()
out = root / 'artifacts'
out.mkdir(exist_ok=True)
archive = out / f'scnet-evidence-{stamp}.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for rel, data in content.items():
        info = tarfile.TarInfo(rel)
        info.size, info.mode = len(data), 0o644
        tf.addfile(info, io.BytesIO(data))
with tarfile.open(archive) as tf:
    for item in manifest:
        assert hashlib.sha256(tf.extractfile(item['path']).read()).hexdigest() == item['sha256']
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': len(content)}, indent=2))
