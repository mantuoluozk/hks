import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
model = Path(os.environ['MODEL_PATH'])
entries = json.loads((root / 'sources/modelscope-files.json').read_text())['Data']['Files']
records=[]
excluded={'README.md','.gitattributes'} if os.environ.get('EXECUTION_PLATFORM')=='SCNet Notebook' else set()
for item in entries:
    if item['Type'] != 'blob':
        continue
    if item['Path'] in excluded:continue
    path = model / item['Path']
    assert path.stat().st_size == item['Size'], item['Path']
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    assert h.hexdigest() == item['Sha256'], item['Path']
    records.append({'file':item['Path'],'bytes':item['Size'],'sha256':h.hexdigest(),'match':True})
    print('SHA256 OK', item['Path'], flush=True)
out=Path(os.environ.get('RUN_RESULTS_DIR',str(root/'results')))
out.mkdir(parents=True,exist_ok=True)
(out/'weights-check.json').write_text(json.dumps({'all_required_files_match':True,'files':records,
    'excluded_non_runtime_files':sorted(excluded),'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned'),
    'source':'SCNet AI community clone' if os.environ.get('EXECUTION_PLATFORM')=='SCNet Notebook' else 'publisher snapshot'},indent=2))
