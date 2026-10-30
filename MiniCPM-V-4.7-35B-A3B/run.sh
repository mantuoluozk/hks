#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
python - <<'PYPORT'
import os,socket
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',int(os.environ['PORT'])))==0:
        raise SystemExit('Port already occupied; refusing to start another model service')
PYPORT
# Fail before constructing an engine when files are incomplete or from another revision.
python scripts/check_weights.py "$MODEL_PATH" --output results/runtime/weights-check.json
bash setup-runtime.sh
bash run-scnet.sh python scripts/check_versions.py
exec bash run-scnet.sh python -m vllm.entrypoints.openai.api_server \
 --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
 --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 2 --dtype bfloat16 \
 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
 --mamba-ssm-cache-dtype float32 \
 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enforce-eager \
 --reasoning-parser minicpmv --no-enable-prefix-caching --limit-mm-per-prompt '{"image":8,"video":0}'
