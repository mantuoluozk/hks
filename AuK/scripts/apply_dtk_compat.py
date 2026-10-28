"""Audited import-path adaptation, scoped to this model's Omni checkout."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT/'runtime/sglang-omni'
original = json.loads((ROOT/'sources/sglang-omni-runtime-files.json').read_text())
expected = {r['path']: r['sha256'] for r in original['files']}
old = b'from sglang.srt.arg_groups.model_override_base import resolved_view'
new = b'from sglang_omni.vendor.sglang.resolved_view import resolved_view'
records = []
for name in ['interface.py', 'cuda.py', 'rocm.py', 'xpu.py']:
    rel = 'sglang_omni/platforms/'+name
    path = runtime/rel
    raw = path.read_bytes()
    if old in raw:
        assert hashlib.sha256(raw).hexdigest() == expected[rel], rel
        assert raw.count(old) == 1, rel
        raw = raw.replace(old, new)
        path.write_bytes(raw)
    else:
        assert new in raw, rel
        assert hashlib.sha256(raw.replace(new, old)).hexdigest() == expected[rel], rel
    records.append({'path': rel, 'original_sha256': expected[rel],
                    'sha256': hashlib.sha256(raw).hexdigest()})
rel = 'sglang_omni/vendor/sglang/resolved_view.py'
raw = b'''"""Resolve the real ServerArgs view across the SGLang module rename."""
try:
    from sglang.srt.arg_groups.model_override_base import resolved_view
except ModuleNotFoundError as error:
    if error.name != "sglang.srt.arg_groups.model_override_base":
        raise
    from sglang.srt.arg_groups.overrides import resolved_view
'''
(runtime/rel).write_bytes(raw)
records.append({'path': rel, 'original_sha256': None, 'sha256': hashlib.sha256(raw).hexdigest()})
(ROOT/'sources/dtk-compat-files.json').write_text(json.dumps({
    'scope': 'model-local Omni checkout; installed SGLang unchanged',
    'change': 'use real resolved_view from legacy overrides module when renamed module is absent',
    'files': records}, indent=2)+'\n')
print(json.dumps({'files': len(records), 'global_packages_modified': False}))
