"""Compare each decompressed publication file byte-for-byte to shared originals."""
import hashlib,json
from pathlib import Path
from evidence_io import read_text
ROOT=Path(__file__).resolve().parents[1];items=json.loads((ROOT/'sources/evidence-view-map.json').read_text())
for r in items:
    p=ROOT/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['publication_sha256']
    original=Path(r['original_path']).read_bytes();restored=read_text(p).encode('utf-8')
    assert original==restored and hashlib.sha256(original).hexdigest()==r['original_sha256']
(ROOT/'sources/evidence-view-validation.json').write_text(json.dumps({'passed':True,'files_checked':len(items),'byte_exact_restoration':True,'model_requests_sent':0},indent=2)+'\n');print('BYTE_EXACT_FILES_VERIFIED',len(items))
