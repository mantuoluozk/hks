"""Verify local publisher files without downloading or loading tensors."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = Path('/root/private_data/SothisAI/model/Aihub/AuK/main/AuK')
out = ROOT / 'results/weights-check-20261028'
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((ROOT/'sources/publisher-files.json').read_text())
records = []
for entry in meta['siblings']:
    if entry['rfilename'] not in {'auk_base.safetensors', 'vae.safetensors', 'config.yaml', 'LICENSE'}:
        continue
    path = base / entry['rfilename']
    record = {'path': str(path), 'exists': path.is_file(), 'expected_bytes': entry['size']}
    if path.is_file():
        record['bytes'] = path.stat().st_size
        digest = hashlib.sha256() if 'lfs' in entry else hashlib.sha1()
        if 'lfs' not in entry:
            digest.update(f'blob {record["bytes"]}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        record['actual_hash'] = digest.hexdigest()
        record['expected_hash'] = entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId']
        record['passed'] = record['bytes'] == entry['size'] and record['actual_hash'] == record['expected_hash']
    else:
        record['passed'] = False
    records.append(record)
result = {'publisher_revision': meta['sha'], 'files': records,
          'main_and_vae_passed': len(records) == 4 and all(r['passed'] for r in records),
          'conditioner_verified': False, 'model_loaded': False}
(out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['main_and_vae_passed'] else 2)
