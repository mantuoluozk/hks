"""Read the two-card environment without loading model weights or changing packages."""
import importlib.metadata as metadata
import json, os, platform, shutil
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
result = {
    'platform': 'SCNet Notebook',
    'python': platform.python_version(),
    'packages': {name: metadata.version(name) for name in
                 ['torch', 'torchvision', 'transformers', 'tokenizers', 'vllm', 'sglang']},
    'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES'),
    'torch_hip': torch.version.hip,
    'device_count': torch.cuda.device_count(),
    'devices': [],
    'shared_storage': str(root),
    'disk_bytes': dict(zip(['total', 'used', 'free'], shutil.disk_usage(root))),
}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    with torch.cuda.device(i):
        a = torch.arange(16, dtype=torch.float32, device=f'cuda:{i}')
        torch.cuda.synchronize()
        value = a.sum().item()
    result['devices'].append({'logical_index': i, 'name': p.name,
                              'memory_bytes': p.total_memory,
                              'architecture': getattr(p, 'gcnArchName', None),
                              'tensor_sum': value, 'tensor_test_passed': value == 120})
result['peer_access'] = (torch.cuda.can_device_access_peer(0, 1)
                         if result['device_count'] == 2 else None)
for name in ['memory.max', 'memory/memory.limit_in_bytes', 'cpu.max', 'cpu/cpu.cfs_quota_us',
             'cpu/cpu.cfs_period_us']:
    p = Path('/sys/fs/cgroup') / name
    if p.exists():
        result[name] = p.read_text().strip()
result['two_device_tensor_test_passed'] = (result['device_count'] == 2 and
    all(d['tensor_test_passed'] for d in result['devices']))
out = root / 'results/environment-20261030.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
