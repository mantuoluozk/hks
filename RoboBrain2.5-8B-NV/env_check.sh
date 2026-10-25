#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p logs results cache "$RUN_RESULTS_DIR"
{
    date -Is
    printf 'model: %s\nimage: %s\ncard: %s\nprecision: %s\nHIP_VISIBLE_DEVICES: %s\n' \
        "$MODEL_ID" "$IMAGE" "$CARD" "$PRECISION" "$HIP_VISIBLE_DEVICES"
    printf 'DTK symlink: '; readlink -f /opt/dtk || true
    for f in /opt/dtk/.info/version /opt/dtk/version; do
        if [ -f "$f" ]; then cat "$f"; fi
    done
    hy-smi --showdriverversion
    python - <<'PY'
import importlib.metadata as m
import json, os, platform, torch
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
        'packages':versions, 'official_scnet_validation':False,'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned')}
print(json.dumps(info, indent=2))
Path(os.environ.get('RUN_RESULTS_DIR','results'),'environment.json').write_text(json.dumps(info, indent=2))
PY
    pip freeze
} 2>&1 | tee env.log
