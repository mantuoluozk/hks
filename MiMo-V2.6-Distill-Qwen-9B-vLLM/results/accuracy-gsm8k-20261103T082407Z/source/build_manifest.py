"""Hash delivery inputs; exclude the manifest and generated verifier output."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[]
mutable={'env.log','server.log','smoke.log','results/server-current.json','results/formal-current.json',
         'bench/round1.jsonl','bench/round2.jsonl','bench/summary.json','bench/evidence-index.json'}
for p in sorted(ROOT.rglob('*')):
    rel=p.relative_to(ROOT)
    if not p.is_file() or any(s in ['artifacts','cache','weights','__pycache__','.git'] for s in rel.parts):continue
    if rel.parts[0]=='data' and rel.as_posix()!='data/README.md':continue
    if p.name in ['submission-manifest.json','verify_result.json'] or p.suffix=='.pyc':continue
    if rel.as_posix() in mutable:continue
    files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'submission-manifest.json').write_text(json.dumps({'scope':'Immutable delivery evidence and code; live outputs and pointers regenerate on replay','mutable_outputs':sorted(mutable),'files':files},indent=2)+'\n',encoding='utf-8')
print('MANIFEST_FILES',len(files))
