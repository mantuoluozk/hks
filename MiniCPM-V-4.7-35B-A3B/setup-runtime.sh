#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python scripts/build_overlay.py
python scripts/fetch_runtime.py
python - <<'PYINSTALL'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path.cwd(); wheels=[]
for row in json.loads((root/'sources/dependency-wheels.json').read_text()):
    p=root/'cache/wheels'/row['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'], p
    wheels.append(str(p))
target=root/'cache/runtime'
stamp=target/'hks-install.json'
locked=json.loads((root/'sources/dependency-wheels.json').read_text())
if stamp.exists() and json.loads(stamp.read_text())==locked:
    print('Pinned runtime already installed');sys.exit(0)
previous=json.loads(stamp.read_text()) if stamp.exists() else []
pending=[str(root/'cache/wheels'/r['file']) for r in locked if r not in previous]
if pending:
    subprocess.run([sys.executable,'-m','pip','install','--no-compile','--no-index','--no-deps','--upgrade','--target',str(target),*pending],check=True)
stamp.write_text(json.dumps(locked,indent=2)+'\n')
PYINSTALL
