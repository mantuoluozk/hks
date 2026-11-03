"""Restore compact evidence JSON values into a new directory (not original bytes)."""
import base64,hashlib,json,sys
from pathlib import Path
from evidence_codec import decode,read_text
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]);assert not OUT.exists();OUT.mkdir(parents=True)
def restore(value):
    if isinstance(value,dict) and value.get('encoding')=='cyclic-range-v1':return decode(value)
    if isinstance(value,dict) and set(value)=={'data_url_header','asset','sha256'}:
        p=ROOT/value['asset'];assert p.resolve().is_relative_to(ROOT)
        blob=p.read_bytes();assert hashlib.sha256(blob).hexdigest()==value['sha256']
        return value['data_url_header']+','+base64.b64encode(blob).decode()
    if isinstance(value,dict):return {k:restore(v) for k,v in value.items()}
    if isinstance(value,list):return [restore(v) for v in value]
    return value
for item in json.loads((ROOT/'sources/evidence-view-map.json').read_text()):
    rel=Path(item['path']);src=ROOT/rel;dst=OUT/rel
    assert src.resolve().is_relative_to(ROOT) and dst.resolve().is_relative_to(OUT.resolve())
    assert hashlib.sha256(src.read_bytes()).hexdigest()==item['publication_sha256']
    if item.get('encoding')=='jsonl-shards-v1':
        index=json.loads(src.read_text());blob=b''.join((src.parent/entry['file']).read_bytes() for entry in index['shards'])
        assert hashlib.sha256(blob).hexdigest()==item['original_sha256']
        dst=OUT/item['restored_path'];assert dst.resolve().is_relative_to(OUT.resolve())
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(blob);continue
    text=read_text(src);dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix=='.jsonl':content=''.join(json.dumps(restore(json.loads(s)),ensure_ascii=False)+'\n' for s in text.splitlines() if s.strip())
    else:content=json.dumps(restore(json.loads(text)),ensure_ascii=False,indent=2)+'\n'
    dst.write_text(content,encoding='utf-8')
print('Restored all compact JSON values; formatting can differ from original bytes')
