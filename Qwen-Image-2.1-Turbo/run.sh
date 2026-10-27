#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash run-scnet.sh python scripts/serve_sglang.py \
  --model-path "${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Qwen-Image-2.1-Turbo/main/Qwen-Image-2.1-Turbo}" \
  --backend diffusers --num-gpus 1 --dit-precision bf16 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30003}" "$@"
