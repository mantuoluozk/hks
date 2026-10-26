#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source env.sh
mkdir -p "$RUN_RESULTS_DIR" logs cache
python scripts/check_weights.py > "$RUN_RESULTS_DIR/startup-weights-check.log"
python - <<'PY'
import importlib.metadata as md
from pathlib import Path
for line in Path('requirements-extra.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    name,version=line.split('==');assert md.version(name)==version,(name,version)
PY
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export MEM_FRAC="${MEM_FRAC:-0.4}"
export MAX_LEN="${MAX_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
python - <<'PY'
import json,os
from pathlib import Path
keys=['MODEL_ID','MODEL_PATH','SERVED_NAME','IMAGE','EXPECTED_IMAGE_ID','HIP_VISIBLE_DEVICES','PORT','ENFORCE_EAGER','MEM_FRAC','MAX_LEN','MAX_NUM_SEQS','EXECUTION_PLATFORM','CARD','ASR_EXTRA_DIR']
Path(os.environ['RUN_RESULTS_DIR'],'runtime-config.json').write_text(json.dumps({k:os.environ.get(k) for k in keys},indent=2))
PY
args=(--model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
      --dtype bfloat16 --gpu-memory-utilization "$MEM_FRAC" --max-model-len "$MAX_LEN"
      --max-num-seqs "$MAX_NUM_SEQS" --generation-config vllm
      --no-enable-prefix-caching --mm-processor-cache-gb 0)
if [ "$ENFORCE_EAGER" = 1 ]; then args+=(--enforce-eager); fi
python - <<'PY'
import os,socket
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind((os.environ['HOST'],int(os.environ['PORT'])))
PY
echo "$$" > "$RUN_RESULTS_DIR/server.pid"
exec python -m vllm.entrypoints.openai.api_server "${args[@]}" > "$RUN_RESULTS_DIR/server-$stamp.log" 2>&1
