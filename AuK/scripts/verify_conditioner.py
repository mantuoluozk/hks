"""Verify the downloaded conditioner against pinned publisher metadata."""
import hashlib
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--whisper', action='store_true')
args=parser.parse_args()
meta = json.loads((ROOT/('sources/whisper-large-v3-files.json' if args.whisper else 'sources/conditioner-files.json')).read_text())
base = ROOT/('evaluators/whisper-large-v3' if args.whisper else 'weights/Qwen2.5-Omni-3B')
out = ROOT/('results/whisper-check' if args.whisper else 'results/conditioner-download-20261028')
out.mkdir(parents=True,exist_ok=True)
records = []
for entry in meta['siblings']:
    if Path(entry['rfilename']).suffix not in {'.safetensors', '.json', '.txt'}:
        continue
    if args.whisper and not (entry['rfilename']=='model.safetensors' or ('/' not in entry['rfilename'] and entry['rfilename'].endswith(('.json','.txt')))):
        continue
    path = base/entry['rfilename']
    record = {'path': entry['rfilename'], 'exists': path.is_file(), 'expected_bytes': entry['size']}
    if path.is_file():
        size = path.stat().st_size
        digest = hashlib.sha256() if 'lfs' in entry else hashlib.sha1()
        if 'lfs' not in entry:
            digest.update(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        record.update(bytes=size, actual_hash=digest.hexdigest(),
                      expected_hash=entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId'])
        record['passed'] = size == entry['size'] and record['actual_hash'] == record['expected_hash']
    else:
        record['passed'] = False
    records.append(record)
result = {'revision': meta['sha'], 'files': records, 'passed': bool(records) and all(r['passed'] for r in records), 'model_loaded': False}
(out/'weights-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['passed'] else 2)
