#!/usr/bin/env bash
# Prepared launcher; full-model inference is not yet verified.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if [[ "${CHECK_WEIGHTS_ON_START:-0}" == "1" ]]; then
  python scripts/check_weights.py "$MODEL_PATH" --output results/runtime/weights-check.json
fi
python - "$PORT" <<'PYPORT'
import socket,sys
with socket.socket() as s:
    s.bind(('127.0.0.1',int(sys.argv[1])))
PYPORT
HF_OVERRIDES='{"activation_situ_beta":null,"activation_situ_linear_beta":null,"attn_res_block_size":null,"latent_moe_use_norm":false,"mla_use_output_gate":false,"routed_expert_hidden_size":null}'
EP_ARGS=()
if [[ "${ENABLE_EXPERT_PARALLEL:-1}" == "1" ]]; then
  EP_ARGS+=(--enable-expert-parallel)
fi
OFFLOAD_ARGS=()
if [[ "${CPU_OFFLOAD_GB:-0}" != "0" ]]; then
  OFFLOAD_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
exec bash run-scnet.sh python -m vllm.entrypoints.openai.api_server   --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" --trust-remote-code   --hf-overrides "$HF_OVERRIDES"   --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --dtype bfloat16   --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"   --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enforce-eager   --no-enable-prefix-caching "${EP_ARGS[@]}" "${OFFLOAD_ARGS[@]}"
