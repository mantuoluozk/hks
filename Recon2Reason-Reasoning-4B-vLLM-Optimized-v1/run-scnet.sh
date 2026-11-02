#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ -f /opt/recon2reason-vllm/platform/env-scnet.sh ]; then
  source /opt/recon2reason-vllm/platform/env-scnet.sh
else
  source /root/private_data/hks/platform/scnet-notebook/env-scnet.sh
fi
source ./env.sh
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$@"
