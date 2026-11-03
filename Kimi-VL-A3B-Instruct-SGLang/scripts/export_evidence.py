"""Create a compact publication view, leaving every original runtime record intact.

Repeated inline images are stored once by content hash. Large generated benchmark
image assets stay at their original shared-storage paths and are indexed explicitly.
"""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts'/time.strftime('publication-%Y%m%dT%H%M%SZ',time.gmtime())
assert not OUT.exists(), 'Use a fresh output; do not overwrite previous evidence'
OUT.mkdir(parents=True)
external=[]
transformed=[]
for folder in ['scripts','sources','results','bench']:
    for src in (ROOT/folder).rglob('*'):
        if not src.is_file() or '__pycache__' in src.parts:
            continue
        rel=src.relative_to(ROOT)
        if src.suffix=='.image' or (src.name=='inputs.json' and src.stat().st_size>8*1024*1024):
            external.append({'path':str(src),'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
            continue
        dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        def compact(value, key=None):
            if key=='url' and isinstance(value,str) and value.startswith('data:image/') and ';base64,' in value:
                header,encoded=value.split(',',1);blob=base64.b64decode(encoded)
                h=hashlib.sha256(blob).hexdigest();asset=Path('assets')/(h+'.png')
                (OUT/asset).parent.mkdir(exist_ok=True)
                if not (OUT/asset).exists(): (OUT/asset).write_bytes(blob)
                return {'data_url_header':header,'asset':asset.as_posix(),'sha256':h}
            if isinstance(value,list):return [compact(v) for v in value]
            if isinstance(value,dict):return {k:compact(v,k) for k,v in value.items()}
            return value
        if src.suffix in ['.json','.jsonl']:
            original=src.read_text(encoding='utf-8')
            if 'data:image/' in original:
                if src.suffix=='.jsonl':
                    content=''.join(json.dumps(compact(json.loads(line)),ensure_ascii=False)+'\n' for line in original.splitlines() if line.strip())
                else:content=json.dumps(compact(json.loads(original)),ensure_ascii=False,indent=2)+'\n'
                dst.write_text(content)
                transformed.append({'path':rel.as_posix(),'original_path':str(src),
                                    'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                                    'publication_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
                                    'change':'Replace repeated inline image data by content-hash asset reference'})
                continue
        shutil.copyfile(src,dst)
for src in list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+list(ROOT.glob('*.md'))+[p for p in [ROOT/'.gitignore',ROOT/'.gitattributes'] if p.exists()]:
    shutil.copyfile(src,OUT/src.name)
(OUT/'sources/external-benchmark-assets.json').write_text(json.dumps(external,indent=2))
(OUT/'sources/evidence-view-map.json').write_text(json.dumps(transformed,indent=2))
print(json.dumps({'publication':str(OUT),'external_assets':len(external)}))
