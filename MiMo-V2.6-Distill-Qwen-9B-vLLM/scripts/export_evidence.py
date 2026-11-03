"""Export a lossless compact view; never modify shared original evidence."""
import base64,hashlib,json,shutil,sys,zlib
from pathlib import Path
from evidence_codec import encode
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]);assert not OUT.exists();OUT.mkdir(parents=True)
changed=[]
def compact(value,key=None):
    if key=='prompt' and isinstance(value,list):
        encoded=encode(value)
        if isinstance(encoded,dict):return encoded
    if key=='url' and isinstance(value,str) and value.startswith('data:image/') and ';base64,' in value:
        header,b64=value.split(',',1);blob=base64.b64decode(b64,validate=True);h=hashlib.sha256(blob).hexdigest()
        asset=Path('assets')/(h+'.png');(OUT/asset).parent.mkdir(exist_ok=True);(OUT/asset).write_bytes(blob)
        return {'data_url_header':header,'asset':asset.as_posix(),'sha256':h}
    if isinstance(value,list):return [compact(x) for x in value]
    if isinstance(value,dict):return {k:compact(v,k) for k,v in value.items()}
    return value
for src in ROOT.rglob('*'):
    rel=src.relative_to(ROOT)
    if not src.is_file() or any(p in ['artifacts','cache','weights','__pycache__','.git'] for p in rel.parts):continue
    if rel.parts[0]=='data' and rel.as_posix()!='data/README.md':continue
    if src.name in ['submission-manifest.json','verify_result.json'] or src.suffix=='.pyc':continue
    dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    blob=src.read_bytes()
    if src.name=='predictions.jsonl' and len(blob)>4*1024**2:
        chunks=[];pending=[];size=0
        for line in blob.splitlines(keepends=True):
            if pending and size+len(line)>4*1024**2:chunks.append(b''.join(pending));pending=[];size=0
            pending.append(line);size+=len(line)
        if pending:chunks.append(b''.join(pending))
        shards=[]
        for i,chunk in enumerate(chunks):
            part=dst.with_name(f'predictions.part{i:03d}.jsonl');part.write_bytes(chunk)
            shards.append({'file':part.name,'sha256':hashlib.sha256(chunk).hexdigest(),'bytes':len(chunk)})
        index=dst.with_suffix('.jsonl.index.json')
        index.write_text(json.dumps({'encoding':'jsonl-shards-v1','sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'shards':shards},indent=2)+'\n')
        changed.append({'path':index.relative_to(OUT).as_posix(),'restored_path':rel.as_posix(),'original_path':str(src),'original_sha256':hashlib.sha256(blob).hexdigest(),'publication_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'encoding':'jsonl-shards-v1','change':'Byte-exact JSONL shards; concatenate in index order to recover original bytes'})
        continue
    if src.suffix in ['.json','.jsonl'] and (b'"prompt": [' in blob or b'data:image/' in blob):
        original=blob.decode('utf-8')
        if src.suffix=='.jsonl':new=''.join(json.dumps(compact(json.loads(line)),ensure_ascii=False)+'\n' for line in original.splitlines() if line.strip())
        else:new=json.dumps(compact(json.loads(original)),ensure_ascii=False,indent=2)+'\n'
        newblob=new.encode('utf-8');dst.write_bytes(newblob)
        changed.append({'path':rel.as_posix(),'original_path':str(src),'original_bytes':len(blob),'original_sha256':hashlib.sha256(blob).hexdigest(),'publication_sha256':hashlib.sha256(newblob).hexdigest(),'change':'Lossless cyclic token-ID representation and/or image content-hash references; response and timing values retained'})
    else:dst.write_bytes(blob)
# Compress and content-deduplicate larger evidence documents. Every stub is read
# transparently by the supplied CPU checker; no original data values are dropped.
for dst in list(OUT.rglob('*')):
    if not dst.is_file() or dst.suffix not in ['.json','.jsonl'] or dst.stat().st_size<=512*1024:continue
    blob=dst.read_bytes();h=hashlib.sha256(blob).hexdigest();compressed=zlib.compress(blob,9);assets=[]
    for i in range(0,len(compressed),2*1024**2):
        chunk=compressed[i:i+2*1024**2];relasset=Path('assets/evidence')/(h+f'.part{i//(2*1024**2):02d}.zlib')
        p=OUT/relasset;p.parent.mkdir(exist_ok=True,parents=True)
        if not p.exists():p.write_bytes(chunk)
        assets.append({'path':relasset.as_posix(),'sha256':hashlib.sha256(chunk).hexdigest()})
    dst.write_text(json.dumps({'encoding':'file-zlib-v1','bytes':len(blob),'sha256':h,'assets':assets},indent=2)+'\n')
    rel=dst.relative_to(OUT);entry=next((x for x in changed if x['path']==rel.as_posix()),None)
    if entry is None:
        original=ROOT/rel;entry={'path':rel.as_posix(),'original_path':str(original),'original_bytes':original.stat().st_size,'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest()};changed.append(entry)
    entry.update(expanded_view_sha256=h,publication_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),file_encoding='file-zlib-v1',change=entry.get('change','Original document values retained')+'; byte-exact zlib file reference, content-deduplicated')
(OUT/'sources/evidence-view-map.json').write_text(json.dumps(changed,indent=2)+'\n')
print(json.dumps({'publication':str(OUT),'transformed_files':len(changed),'files':sum(p.is_file() for p in OUT.rglob('*')),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
