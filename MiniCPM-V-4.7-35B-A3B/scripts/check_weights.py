"""Check local runtime files against the fixed publisher manifest; never download."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('weights', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((Path(__file__).resolve().parents[1] / 'sources/publisher-files.json').read_text(encoding='utf-8'))
previous = json.loads(a.output.read_text(encoding='utf-8')) if a.output.exists() else {}
prior = {r['file']: r for r in previous.get('files', [])} if previous.get('revision') == manifest['sha'] else {}
rows = []
for item in manifest['siblings']:
    name = item['rfilename']
    if '/' in name or not name.endswith(('.json', '.jinja', '.safetensors')):
        continue
    f = a.weights / name
    row = {'file': name, 'exists': f.is_file(), 'ok': False}
    if f.is_file():
        st = f.stat()
        identity = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        row['identity'] = identity
        cached = prior.get(name, {})
        if cached.get('ok') and cached.get('identity') == identity and cached.get('sha256') and cached.get('source_resolved') == str(f.resolve()):
            row.update(cached); row['reused_unchanged_file'] = True; rows.append(row); continue
        row['source_resolved'] = str(f.resolve())
        size = st.st_size
        row['size'] = size
        if size == item['size']:
            sha = hashlib.sha256()
            blob = hashlib.sha1(f'blob {size}\0'.encode())
            with f.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    sha.update(chunk)
                    if 'lfs' not in item: blob.update(chunk)
            after=f.stat()
            assert identity==[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns], f'File changed during hashing: {f}'
            row['sha256'] = sha.hexdigest()
            row['ok'] = (sha.hexdigest() == item['lfs']['sha256'] if 'lfs' in item else blob.hexdigest() == item['blobId'])
    rows.append(row)
result = {'revision': manifest['sha'], 'weights_path': str(a.weights), 'files': rows, 'passed': bool(rows) and all(r['ok'] for r in rows)}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': result['passed'], 'checked': len(rows), 'missing_or_invalid': [r['file'] for r in rows if not r['ok']]}))
raise SystemExit(0 if result['passed'] else 2)
