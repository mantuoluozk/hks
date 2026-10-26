#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs results cache
mkdir -p "$RUN_RESULTS_DIR"
python scripts/check_weights.py
python - <<'PY'
import os,socket
with socket.socket() as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 s.bind((os.environ['HOST'],int(os.environ['PORT'])))
PY
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/server-$RUN_ID.log"
if [ "$RUN_RESULTS_DIR" = results ]; then
 ln -sfn "$LOG" server.log
else
 LOG="$RUN_RESULTS_DIR/server-$RUN_ID.log"
 ln -sfn "$(basename "$LOG")" "$RUN_RESULTS_DIR/server.log"
fi
CMD=(python -m sglang.launch_server
 --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME"
 --host "$HOST" --port "$PORT" --tp-size "$TP" --dtype bfloat16
 --context-length "$MAX_LEN" --mem-fraction-static "$MEM_FRAC"
 --max-running-requests "$MAX_RUNNING_REQUESTS" --chunked-prefill-size 4096
 --page-size 1
 --attention-backend "$ATTENTION_BACKEND" --mm-attention-backend sdpa
 --reasoning-parser mimo
 --cuda-graph-backend-decode "$DECODE_GRAPH_BACKEND" --cuda-graph-backend-prefill disabled
 --cuda-graph-max-bs-decode "$DECODE_GRAPH_MAX_BS")
case "$DISABLE_OVERLAP_SCHEDULE" in
 1) CMD+=(--disable-overlap-schedule) ;;
 0) ;;
 *) echo 'DISABLE_OVERLAP_SCHEDULE must be 0 or 1' >&2; exit 2 ;;
esac
printf '%q ' "${CMD[@]}" | tee "$LOG"
printf '\n' | tee -a "$LOG"
"${CMD[@]}" >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_RESULTS_DIR/server.pid"
export SERVER_RUN_ID="$RUN_ID"
python - <<'PY'
import os,json
from pathlib import Path
keys=['MODEL_ID','MODEL_PATH','SERVED_NAME','IMAGE','CARD','PRECISION','HIP_VISIBLE_DEVICES','HOST','PORT','TP','MAX_LEN','MEM_FRAC','MAX_RUNNING_REQUESTS','ATTENTION_BACKEND','SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS','SGLANG_KV_LAYOUT_HCU_FA','DECODE_GRAPH_BACKEND','DECODE_GRAPH_MAX_BS','DISABLE_OVERLAP_SCHEDULE','SERVER_RUN_ID','EXECUTION_PLATFORM']
Path(os.environ['RUN_RESULTS_DIR'],'runtime-config.json').write_text(json.dumps({k:os.environ.get(k) for k in keys},indent=2))
PY
for ((t=0;t<LOAD_TIMEOUT;t+=5)); do
 if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Server exited. See $LOG" >&2
  wait "$SERVER_PID"
  exit 1
 fi
 if curl --noproxy '*' -fsS --max-time 4 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "Ready after ${t}s; PID=$SERVER_PID; log=$LOG"
  wait "$SERVER_PID"
  exit $?
 fi
 sleep 5
done
echo "Readiness timeout; inspect $LOG and results/server.pid" >&2
exit 1
