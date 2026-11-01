#!/usr/bin/env bash
# Default launcher for Kimi-Linear-48B-A3B-Instruct on the SCNet four-card image.
# Starts the vLLM TP4 service used for every result in this directory.
#
# IMPORTANT: the base image does NOT contain the KDA runtime fix. A fresh
# instance reproduces the degenerate "!!!!" output until the patch is applied.
# Run this once per new container before starting:
#     bash run-scnet.sh python runtime/apply_vllm_kda_fix.py
#     bash run-scnet.sh python runtime/apply_vllm_kda_fix.py --check
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source ./env.sh

RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-$PWD/results/preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_RESULTS_DIR"

# Refuse to start on an unpatched installation: the service would load and then
# emit only "!" tokens, wasting a full weight-load cycle.
if ! bash run-scnet.sh python runtime/apply_vllm_kda_fix.py --check > "$RUN_RESULTS_DIR/patch-check.log" 2>&1; then
  echo "ERROR: the Kimi KDA runtime fix is not applied to the installed packages." >&2
  echo "       Run: bash run-scnet.sh python runtime/apply_vllm_kda_fix.py" >&2
  echo "       Details: $RUN_RESULTS_DIR/patch-check.log" >&2
  echo "       (idempotent; saves originals as *.kimi-linear-original)" >&2
  exit 3
fi

if [[ "${CHECK_WEIGHTS_ON_START:-1}" == "1" ]]; then
  bash run-scnet.sh python scripts/check_weights.py "$MODEL_PATH" \
    --output "$RUN_RESULTS_DIR/weights-check.json"
fi

mkdir -p results
echo "$$" > results/server.pid

exec bash run-scnet.sh python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --trust-remote-code \
  --no-enable-prefix-caching
