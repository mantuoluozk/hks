"""Index exact publication bytes; do not include weights, datasets, or caches."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve()
excluded={'submission-manifest.json'}
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.relative_to(root).as_posix() in excluded:continue
    rel=p.relative_to(root)
    assert not set(rel.parts).intersection({'weights','data','cache','artifacts','__pycache__','.git'}),rel
    assert not p.is_symlink(),rel
    files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'submission-manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n',encoding='utf-8')
print('Manifest files',len(files))
