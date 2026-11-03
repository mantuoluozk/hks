"""Immutable delivery byte hashes; live replay outputs are explicitly excluded."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
mutable={'env.log','server.log','smoke.log','results/server-current.json','results/formal-current.json','results/bench-current.json','results/accuracy-current.json','results/official-current.json'}
mutable.update('bench/'+x for x in ['round1.jsonl','round2.jsonl','summary.json','requests.jsonl','warmups.jsonl','independent-check.json','evidence-index.json'])
files=[]
for p in sorted(ROOT.rglob('*'),key=lambda p:p.relative_to(ROOT).as_posix()):
    rel=p.relative_to(ROOT)
    if any(x in ['cache','weights','artifacts','__pycache__','.git'] for x in rel.parts):continue
    if rel.parts[0]=='data' and rel.as_posix()!='data/README.md':continue
    if not p.is_file() or p.suffix=='.pyc' or rel.as_posix() in mutable or p.name in ['submission-manifest.json','verify_result.json']:continue
    files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'submission-manifest.json').write_text(json.dumps({'scope':'Immutable source and saved evidence; live replay aliases excluded, generated verifier output excluded to avoid self-reference','mutable_outputs':sorted(mutable),'files':files},indent=2)+'\n',encoding='utf-8',newline='\n');print('MANIFEST_INPUTS',len(files))
