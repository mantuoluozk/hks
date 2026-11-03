"""Byte-exact compressed publication view; originals remain unchanged."""
import hashlib,json,sys,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(sys.argv[1]);assert not OUT.exists();OUT.mkdir(parents=True)
changed=[]
for src in ROOT.rglob('*'):
    rel=src.relative_to(ROOT)
    if any(x in ['cache','weights','artifacts','__pycache__','.git'] for x in rel.parts):continue
    if rel.parts[0]=='data' and rel.as_posix()!='data/README.md':continue
    if not src.is_file() or src.suffix=='.pyc' or src.name in ['submission-manifest.json','verify_result.json']:continue
    dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);blob=src.read_bytes()
    if src.suffix in ['.json','.jsonl'] and len(blob)>512*1024:
        h=hashlib.sha256(blob).hexdigest();compressed=zlib.compress(blob,9);assets=[]
        for i in range(0,len(compressed),2*1024**2):
            part=compressed[i:i+2*1024**2];path=Path('assets/evidence')/(h+f'.part{i//(2*1024**2):02d}.zlib');asset=OUT/path;asset.parent.mkdir(parents=True,exist_ok=True)
            if not asset.exists():asset.write_bytes(part)
            assets.append({'path':path.as_posix(),'sha256':hashlib.sha256(part).hexdigest()})
        dst.write_text(json.dumps({'encoding':'file-zlib-v1','bytes':len(blob),'sha256':h,'assets':assets},indent=2)+'\n')
        changed.append({'path':rel.as_posix(),'original_path':str(src),'original_bytes':len(blob),'original_sha256':h,'publication_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'encoding':'file-zlib-v1'})
    else:dst.write_bytes(blob)
(OUT/'sources/evidence-view-map.json').write_text(json.dumps(changed,indent=2)+'\n');print('BYTE_EXACT_PUBLICATION_FILES',sum(p.is_file() for p in OUT.rglob('*')))
