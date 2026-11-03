"""Rebuild the model-local overlay only from the exact reviewed vendor source."""
import hashlib
import json
from pathlib import Path
import sglang

root=Path(__file__).resolve().parents[1]
original=Path(sglang.__file__).parent/'srt/models/kimi_vl.py'
expected=(root/'sources/kimi_vl.py').read_bytes()
assert original.read_bytes()==expected, 'Installed Kimi source differs from reviewed source'
needle='        self.act = GELUActivation()\n        self.act = QuickGELU()'
source=expected.decode()
assert source.count(needle)==1
patched=source.replace(needle,'        self.act = GELUActivation()')
(root/'sources/kimi_vl.patched.py').write_text(patched)
print(json.dumps({'source_sha256':hashlib.sha256(expected).hexdigest(),
                  'patched_sha256':hashlib.sha256(patched.encode()).hexdigest()}))
