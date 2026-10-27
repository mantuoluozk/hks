"""Install only the reviewed wheels into this model's shared runtime directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'sources/runtime-dependencies.json').read_text())
wheels=[]
for item in manifest['packages']:
    if not item['filename'].endswith('.whl'):continue
    path=ROOT/'runtime/wheels'/item['filename']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    wheels.append(str(path))
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(ROOT/'runtime/site-packages'),*wheels],check=True)
