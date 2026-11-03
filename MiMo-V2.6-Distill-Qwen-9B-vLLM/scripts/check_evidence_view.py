"""Compare every compact publication JSON value against shared original evidence."""
import base64,hashlib,json
from pathlib import Path
from evidence_codec import decode,read_text
ROOT=Path(__file__).resolve().parents[1]
def restore(v):
    if isinstance(v,dict) and v.get('encoding')=='cyclic-range-v1':return decode(v)
    if isinstance(v,dict) and set(v)=={'data_url_header','asset','sha256'}:
        p=ROOT/v['asset'];assert p.resolve().is_relative_to(ROOT)
        b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==v['sha256']
        return v['data_url_header']+','+base64.b64encode(b).decode()
    if isinstance(v,dict):return {k:restore(x) for k,x in v.items()}
    if isinstance(v,list):return [restore(x) for x in v]
    return v
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
checked=[]
for entry in json.loads((ROOT/'sources/evidence-view-map.json').read_text()):
    original=Path(entry['original_path']).read_bytes();assert hashlib.sha256(original).hexdigest()==entry['original_sha256']
    p=ROOT/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['publication_sha256']
    if entry.get('encoding')=='jsonl-shards-v1':
        index=json.loads(read_text(p));blob=b''.join((p.parent/x['file']).read_bytes() for x in index['shards'])
        assert blob==original;checked.append(entry['path']);continue
    text=read_text(p)
    if p.suffix=='.jsonl':
        old=original.decode().splitlines();new=text.splitlines();assert len(old)==len(new)
        for a,b in zip(old,new):assert canonical(json.loads(a))==canonical(restore(json.loads(b))),entry['path']
    else:assert canonical(json.loads(original))==canonical(restore(json.loads(text))),entry['path']
    checked.append(entry['path'])
(ROOT/'sources/evidence-view-validation.json').write_text(json.dumps({'passed':True,'files_checked':len(checked),'all_original_json_values_preserved':True,'scope':'CPU lossless export validation; not new model inference','files':checked},indent=2)+'\n')
print('LOSSLESS_VIEW_FILES_VERIFIED',len(checked))
