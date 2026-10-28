#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export AUK_MODEL_PATH="${AUK_MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/AuK/main/AuK}"
export AUK_ENCODER_PATH="${AUK_ENCODER_PATH:-$PWD/weights/Qwen2.5-Omni-3B}"
export AUK_ATTENTION_BACKEND="${AUK_ATTENTION_BACKEND:-math}"
exec bash ./run-scnet.sh python scripts/serve.py serve \
  --model-path "$AUK_MODEL_PATH" --host 127.0.0.1 --port "${AUK_PORT:-30005}" \
  --conditioning.factory.text_encoder_path "$AUK_ENCODER_PATH" \
  --conditioning.factory.max_batch_size 1 \
  --conditioning.factory.dtype bfloat16 \
  --auk_engine.factory.max_batch_size 1 \
  --auk_engine.factory.dtype "${AUK_DIT_DTYPE:-float32}" \
  --decode.factory.max_batch_size 1 \
  "$@"
