"""Read original predictions or byte-exact publication shards."""
import hashlib,json
from pathlib import Path
from evidence_codec import read_text
def read_predictions(folder):
    p=Path(folder)/'predictions.jsonl'
    if p.exists():blob=read_text(p).encode('utf-8')
    else:
        index=json.loads(p.with_suffix('.jsonl.index.json').read_text());chunks=[]
        assert index['encoding']=='jsonl-shards-v1'
        for entry in index['shards']:
            part=p.parent/entry['file'];assert part.resolve().is_relative_to(p.parent.resolve())
            data=part.read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256'];chunks.append(data)
        blob=b''.join(chunks);assert hashlib.sha256(blob).hexdigest()==index['sha256'] and len(blob)==index['bytes']
    return [json.loads(s) for s in blob.decode('utf-8').splitlines() if s.strip()]
