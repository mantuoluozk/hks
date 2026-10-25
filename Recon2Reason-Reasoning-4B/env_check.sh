#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs results cache
if [ "${EXECUTION_PLATFORM:-}" = 'SCNet Notebook' ]; then
    mkdir -p repro/self-owned-prevalidation/results
    if [ -f env.log ] && [ ! -f repro/self-owned-prevalidation/env.log ]; then cp env.log repro/self-owned-prevalidation/env.log; fi
    if [ -f results/environment.json ] && [ ! -f repro/self-owned-prevalidation/results/environment.json ]; then cp results/environment.json repro/self-owned-prevalidation/results/environment.json; fi
fi
{
    date -Is
    printf 'model: %s\nimage: %s\ncard: %s\nprecision: %s\nHIP_VISIBLE_DEVICES: %s\n' \
        "$MODEL_ID" "$IMAGE" "$CARD" "$PRECISION" "$HIP_VISIBLE_DEVICES"
    printf 'DTK symlink: '; readlink -f /opt/dtk || true
    for f in /opt/dtk/.info/version /opt/dtk/version; do
        if [ -f "$f" ]; then cat "$f"; fi
    done
    if [ "${EXECUTION_PLATFORM:-}" = 'SCNet Notebook' ]; then hy-smi --showdriverversion; else hy-smi; fi
    python - <<'PY'
import importlib.metadata as m
import json, platform, torch
from pathlib import Path
from datetime import datetime, timezone
packages = ['torch','torchvision','torchaudio','triton','sglang','sglang-kernel',
            'transformers','tokenizers','flash_attn','aiter','vllm','vllm-hcu','modelscope','modelscope-hub','pillow']
versions = {}
for name in packages:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
info = {'captured_at':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
        'hip':torch.version.hip, 'visible_device_count':torch.cuda.device_count(),
        'devices':[{'name':torch.cuda.get_device_name(i),
                    'properties':str(torch.cuda.get_device_properties(i))} for i in range(torch.cuda.device_count())],
        'packages':versions, 'execution_platform':__import__('os').environ.get('EXECUTION_PLATFORM','self-owned BW100'), 'official_scnet_validation':False}
print(json.dumps(info, indent=2))
Path('results/environment.json').write_text(json.dumps(info, indent=2))
PY
    pip freeze
} 2>&1 | tee env.log

