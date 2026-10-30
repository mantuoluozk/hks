"""Fetch only the two small pinned dependency wheels, never model files."""
import hashlib, json, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for row in json.loads((root/'sources/dependency-wheels.json').read_text()):
    p=root/'cache/wheels'/row['file'];p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']:continue
    with urllib.request.urlopen(row['url'],timeout=60) as r:raw=r.read()
    assert len(raw)==row['size'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['file']
    tmp=p.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(p)
    print('Verified dependency:',row['file'])
