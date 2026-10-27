"""Verify every runtime checkpoint file against publisher SHA256 metadata."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
model=Path(os.environ['MODEL_PATH'])
source=json.loads((ROOT/'sources/modelscope-files.json').read_text())
records=[]
for item in source['Data']['Files']:
    name=item['Path']
    if item['Type']!='blob' or name in {'.gitattributes','README.md'} or name.startswith('assets/'):continue
    path=model/name
    assert path.stat().st_size==item['Size'],name
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(16<<20):h.update(block)
    assert h.hexdigest()==item['Sha256'],name
    records.append({'path':name,'bytes':item['Size'],'sha256':h.hexdigest(),'publisher_match':True})
    print('SHA256 OK',name,flush=True)
out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
(out/'weights-check.json').write_text(json.dumps({'all_runtime_files_match':True,'files':records,'source':'SCNet AI community'},indent=2))
