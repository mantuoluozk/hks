"""Offline runtime/weight/source verification; no download or package mutation."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import torch
import sglang

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/preflight'
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT/'sources/publisher-files.json').read_text())
assert manifest['sha'] == '398eede0903cd983a2bfa0cc634e9ac1d843f375'
proof = {}
for f in manifest['siblings']:
    name = f['rfilename']
    if '/' in name or name.startswith('.'):
        continue
    p = ROOT/'weights'/name
    assert p.stat().st_size == f['size'], name
    h = hashlib.sha256() if f.get('lfs') else hashlib.sha1()
    if not f.get('lfs'):
        h.update(f"blob {f['size']}\0".encode())
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    assert h.hexdigest() == (f['lfs']['sha256'] if f.get('lfs') else f['blobId']), name
    proof[name] = {'bytes': p.stat().st_size, 'hash': h.hexdigest(), 'target': str(p.resolve())}
    print('verified', name, flush=True)
(OUT/'weights.json').write_text(json.dumps(proof, indent=2))
base = Path(sglang.__file__).parent/'srt'
for name in ['models/kimi_vl.py', 'models/kimi_vl_moonvit.py', 'models/deepseek_v2.py', 'layers/moe/topk.py']:
    p = base/name
    shutil.copyfile(p, ROOT/'sources'/p.name)
source = (ROOT/'sources/kimi_vl.py').read_text()
needle = '        self.act = GELUActivation()\n        self.act = QuickGELU()'
assert source.count(needle) == 1
publisher = (ROOT/'weights/modeling_kimi_vl.py').read_text()
assert 'self.act = GELUActivation()' in publisher
(ROOT/'sources/kimi_vl.patched.py').write_text(source.replace(needle, '        self.act = GELUActivation()'))
shutil.copyfile(ROOT/'weights/modeling_kimi_vl.py', ROOT/'sources/publisher_modeling_kimi_vl.py')
record = {'packages': {n: metadata.version(n) for n in ['torch','sglang','transformers','tokenizers']},
          'gpu': torch.cuda.get_device_name(0), 'memory': torch.cuda.mem_get_info(),
          'hip': torch.version.hip, 'projector_patch': 'Remove QuickGELU override; preserve publisher GELU',
          'model_inference_verified': False}
(OUT/'environment.json').write_text(json.dumps(record, indent=2))
from sglang.srt.models.kimi_vl import KimiVLForConditionalGeneration
print(json.dumps(record), flush=True)
