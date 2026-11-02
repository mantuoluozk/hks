#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ -f /opt/recon2reason-vllm/platform/env-scnet.sh ]; then
  source /opt/recon2reason-vllm/platform/env-scnet.sh
else
  source /root/private_data/hks/platform/scnet-notebook/env-scnet.sh
fi
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
python scripts/collect_env.py | tee env.log
