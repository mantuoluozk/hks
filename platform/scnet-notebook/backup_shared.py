"""Snapshot explicit project directories on shared storage; never read home secrets."""
import hashlib
import io
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ.get('HKS_ROOT', '/root/private_data/hks')).resolve()
assert Path('/root/private_data').is_mount(), 'Persistent storage must be mounted'
assert root.is_relative_to(Path('/root/private_data').resolve()), root
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
dest = root / 'backups' / stamp
dest.mkdir(parents=True, exist_ok=False)
archive = dest / 'code-evidence.tar.gz'
excluded = {'.git', '.ssh', '.jupyter', 'cache', 'data', 'datasets', 'weights', 'artifacts', '__pycache__', '.venv', 'tmp', 'sglang_storage'}
weight_suffixes = {'.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx', '.parquet', '.arrow'}
records = []
with tarfile.open(archive.with_suffix('.partial'), 'w:gz') as tf:
    for dirname in ['models', 'platform']:
        for base, dirs, files in os.walk(root / dirname):
            dirs[:] = [d for d in dirs if d not in excluded and not (Path(base) / d).is_symlink()]
            for name in sorted(files):
                p = Path(base) / name
                # Native crash dumps can contain whole model weights and process memory.
                if (p.is_symlink() or name.startswith('.') or p.suffix in weight_suffixes
                        or name == 'core' or re.fullmatch(r'core\.\d+', name)
                        or p.suffix.lower() in {'.core', '.dmp'}):
                    continue
                rel = p.relative_to(root).as_posix()
                raw = p.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                info = tf.gettarinfo(str(p), arcname=rel)
                info.size = len(raw)
                tf.addfile(info, io.BytesIO(raw))
                records.append({'path': rel, 'bytes': len(raw), 'sha256': digest})
archive.with_suffix('.partial').rename(archive)
with tarfile.open(archive) as tf:
    assert sorted(tf.getnames()) == sorted(r['path'] for r in records)
    for r in records:
        assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest() == r['sha256']
summary = {'created_at': stamp, 'files': records, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
           'archive_contents_verified': True, 'weights_and_datasets': 'Stored separately on persistent shared storage; excluded from snapshot'}
(dest / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'snapshot': str(dest), 'files': len(records), 'bytes': archive.stat().st_size,
                  'archive_sha256': summary['archive_sha256'], 'verified': True}))
