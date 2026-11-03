#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/runtime-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_RESULTS_DIR
mkdir -p "$RUN_RESULTS_DIR"
ARGS=(--model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$PORT"
 --dtype bfloat16 --tensor-parallel-size 1 --trust-remote-code
 --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192
 --gpu-memory-utilization "$MEM_FRAC" --mamba-ssm-cache-dtype float32
 --reasoning-parser mimo --enable-auto-tool-choice --tool-call-parser mimo
 --no-enable-prefix-caching --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":2,"video":0}'
 --generation-config auto --seed 42 --no-enable-log-requests)
if [ "${EXECUTION_MODE:-eager}" = eager ]; then
 ARGS+=(--enforce-eager)
elif [ "$EXECUTION_MODE" = graph ]; then
 ARGS+=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}')
else
 echo 'EXECUTION_MODE must be eager or graph' >&2; exit 1
fi
exec python scripts/service.py "${ARGS[@]}"
