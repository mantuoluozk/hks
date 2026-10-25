#!/usr/bin/env bash
# Run INSIDE the dedicated container; logs from each launch are preserved.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs results cache
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model-00001-of-00002.safetensors"
test -f "$MODEL_PATH/model-00002-of-00002.safetensors"
python - <<'PY'
import os, socket
s=socket.socket()
try: s.bind((os.environ['HOST'],int(os.environ['PORT'])))
finally: s.close()
PY
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/server-$RUN_ID.log"
ln -sfn "$LOG" server.log
CMD=(python -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME"
    --host "$HOST" --port "$PORT" --tp-size "$TP" --dtype bfloat16
    --context-length "$MAX_LEN" --mem-fraction-static "$MEM_FRAC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --chunked-prefill-size 4096
    --attention-backend triton --mm-attention-backend sdpa
    --cuda-graph-backend-decode "$DECODE_GRAPH_BACKEND" --cuda-graph-backend-prefill disabled
    --cuda-graph-max-bs-decode "$DECODE_GRAPH_MAX_BS"
    --enable-multimodal)
printf '%q ' "${CMD[@]}" | tee "$LOG"
printf '\n' | tee -a "$LOG"
"${CMD[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > results/server.pid
export SERVER_RUN_ID="$RUN_ID"
python - <<'PY'
import json,os
from pathlib import Path
keys=['MODEL_ID','MODEL_PATH','SERVED_NAME','IMAGE','FRAMEWORK','CARD','PRECISION','HIP_VISIBLE_DEVICES',
      'HOST','PORT','TP','MAX_LEN','MEM_FRAC','MAX_RUNNING_REQUESTS','DECODE_GRAPH_BACKEND','DECODE_GRAPH_MAX_BS','SERVER_RUN_ID']
Path('results/runtime-config.json').write_text(json.dumps({k:os.environ[k] for k in keys},indent=2))
PY
for ((t=0; t<LOAD_TIMEOUT; t+=5)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server exited. See $LOG" >&2
        wait "$SERVER_PID"
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 4 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        printf 'Ready after %ss; PID=%s; log=%s\n' "$t" "$SERVER_PID" "$LOG"
        wait "$SERVER_PID"
        exit $?
    fi
    sleep 5
done
echo "Readiness timeout. See $LOG. Process left for diagnosis; no unrelated process is stopped." >&2
exit 1

