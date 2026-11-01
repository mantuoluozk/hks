#!/usr/bin/env bash
# Official-template image matrix and independent service reproduction.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if [ -f /opt/recon2reason-vllm/platform/env-scnet.sh ]; then
  source /opt/recon2reason-vllm/platform/env-scnet.sh
else
  source /root/private_data/hks/platform/scnet-notebook/env-scnet.sh
fi
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/run_formal_matrix.py
