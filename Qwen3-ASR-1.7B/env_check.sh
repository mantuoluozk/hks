#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source env.sh
mkdir -p results logs cache
python scripts/inspect_environment.py | tee env.log
python - <<'PY'
import importlib.metadata as md
from pathlib import Path
for line in Path('requirements-extra.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):continue
    name,version=line.split('==')
    assert md.version(name)==version,(name,version)
print('Pinned audio dependency versions verified.')
PY
