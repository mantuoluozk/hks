"""Require every pinned snapshot file to pass SHA256 before serving."""
import hashlib,json,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=Path(os.environ.get('MODEL_PATH','/data1/model/MiMo-V2.6-Distill-Qwen-9B'))
files=json.loads((root/'scripts/files.json').read_text())['Data']['Files']
records=[]
excluded={'README.md','.gitattributes'} if os.environ.get('EXECUTION_PLATFORM')=='SCNet Notebook' else set()
if (dest/'upstream-source.json').is_file(): excluded.add('configuration.json')
for item in files:
    if item['Name'] in excluded: continue
    p=dest/item['Name']
    assert p.is_file() and p.stat().st_size==item['Size'],f'Missing/incomplete: {p}'
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    assert h.hexdigest()==item['Sha256'],f'SHA256 mismatch: {p}'
    records.append({'file':item['Name'],'bytes':item['Size'],'sha256':h.hexdigest(),'match':True})
out=Path(os.environ.get('RUN_RESULTS_DIR','results'))/'weights-check.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned'),'all_required_files_match':True,'files':records,'excluded_non_runtime_files':sorted(excluded),'weight_source':'publisher Hugging Face snapshot' if (dest/'upstream-source.json').is_file() else 'SCNet AI community clone' if os.environ.get('EXECUTION_PLATFORM')=='SCNet Notebook' else 'upstream snapshot','official_acceptance_passed':None},indent=2)+'\n')
print(f'All {len(records)} pinned runtime files verified',flush=True)
