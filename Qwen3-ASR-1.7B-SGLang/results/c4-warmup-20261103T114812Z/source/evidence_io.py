"""Read evidence, including content-deduplicated byte-exact compressed files."""
import hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read_text(path):
    blob=Path(path).read_bytes()
    if len(blob)<2048:
        try:r=json.loads(blob)
        except (ValueError,UnicodeDecodeError):r=None
        if isinstance(r,dict) and r.get('encoding')=='file-zlib-v1':
            chunks=[]
            for a in r['assets']:
                p=ROOT/a['path'];assert p.resolve().is_relative_to(ROOT)
                b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256'];chunks.append(b)
            blob=zlib.decompress(b''.join(chunks));assert len(blob)==r['bytes'] and hashlib.sha256(blob).hexdigest()==r['sha256']
    return blob.decode('utf-8')
def read_json(path):return json.loads(read_text(path))
def read_jsonl(path):return [json.loads(s) for s in read_text(path).splitlines() if s.strip()]
