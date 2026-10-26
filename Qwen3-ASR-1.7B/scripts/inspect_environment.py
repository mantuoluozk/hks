"""Inspect the pinned DCU image before installing or loading a model."""
import importlib.metadata as md
import importlib.util
import json
import platform
import os
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
result = {'python': platform.python_version(), 'torch': torch.__version__,
          'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned'),
          'image':os.environ.get('IMAGE'),'extra_package_dir':os.environ.get('ASR_EXTRA_DIR'),
          'hip': torch.version.hip, 'packages': {}, 'devices': []}
for package in ['vllm', 'transformers', 'openai', 'soundfile', 'librosa', 'torchaudio', 'qwen-asr', 'soxr', 'av', 'editdistance', 'pyarrow']:
    try:
        result['packages'][package] = md.version(package)
    except md.PackageNotFoundError:
        result['packages'][package] = None
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    result['devices'].append({'logical_device': i, 'name': p.name,
                              'gcn_arch': p.gcnArchName, 'memory': p.total_memory})
spec = importlib.util.find_spec('vllm')
if spec:
    folder = Path(spec.origin).parent
    result['asr_model_sources'] = [str(p.relative_to(folder)) for p in folder.rglob('*qwen*asr*.py')]
    registry = folder / 'model_executor/models/registry.py'
    result['registry_has_qwen3_asr'] = registry.exists() and 'Qwen3ASR' in registry.read_text()
out=Path(os.environ.get('RUN_RESULTS_DIR',str(root/'results')));out.mkdir(parents=True,exist_ok=True)
(out/'environment.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
