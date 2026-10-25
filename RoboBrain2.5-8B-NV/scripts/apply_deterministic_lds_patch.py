"""Pinned HIP deterministic MM/BMM compatibility patch for this base image."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
root = Path(__file__).resolve().parents[1]
spec = importlib.util.find_spec('sglang.srt.batch_invariant_ops.batch_invariant_ops')
path = Path(spec.origin)
raw = path.read_text()
expected_before = '9a9fdb89d84f8c392ae9a46f46c0a846b332e78f4ae77c24f240f8db55e6a6e4'
expected_after = '9668a00e1f8d068d3528c5f6fea55b7ec6994259ebf42fc2edf031525aa87a1b'
marker = '# HKS: bound HIP deterministic BF16 GEMM staging to 64 KiB LDS.'
marker_v2 = '# HKS: bound HIP batch-invariant MM and BMM staging to 64 KiB LDS.'
marker_v3 = '# HKS: use IEEE FP32 inputs for deterministic MM/BMM correctness.'
needle = '    # print(a.device, b.device, c.device)\n    matmul_kernel_persistent[grid]('
addition = '''    # HKS: bound HIP batch-invariant MM and BMM staging to 64 KiB LDS.
    if (torch.version.hip
            and torch.cuda.get_device_properties(a.device).shared_memory_per_block <= 65536):
        configs[dtype]["num_stages"] = 1 if dtype == torch.float16 else 2
    # print(a.device, b.device, c.device)
    matmul_kernel_persistent[grid]('''
if marker_v3 not in raw:
    backup = root / 'cache/deterministic-patch-original.py'
    if marker in raw or marker_v2 in raw:
        raw = backup.read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == expected_before, 'Unexpected framework source; inspect before patching'
    assert raw.count(needle) == 1
    backup.parent.mkdir(exist_ok=True)
    backup.write_text(raw)
    patched = raw.replace(needle, addition)
    bmm_needle = '        config = configs.get(dtype)'
    assert patched.count(bmm_needle) == 1
    patched = patched.replace(bmm_needle, '''        # HKS: limit deterministic attention's FP32/BF16 BMM staging too.
        if (torch.version.hip
                and torch.cuda.get_device_properties(a.device).shared_memory_per_block <= 65536):
            configs[dtype]["num_stages"] = 1 if dtype == torch.float16 else 2
        config = configs.get(dtype)''')
    assert patched.count('accumulator = tl.dot(a, b, accumulator)') == 2
    patched = patched.replace('accumulator = tl.dot(a, b, accumulator)',
        'accumulator = tl.dot(a, b, accumulator, input_precision="ieee")')
    patched = marker_v3 + '\n' + patched
    assert hashlib.sha256(patched.encode()).hexdigest() == expected_after
    path.write_text(patched)
    (Path(os.environ.get('RUN_RESULTS_DIR',str(root/'results'))) / 'deterministic-lds-patch.json').write_text(json.dumps({
        'relative_framework_path': 'sglang/srt/batch_invariant_ops/batch_invariant_ops.py',
        'before_sha256': hashlib.sha256(raw.encode()).hexdigest(),
        'after_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'reason': 'Fit MM/BMM staging to 64 KiB LDS; IEEE FP32 input precision avoids the observed ~7.7e-4 relative FP32 BMM error',
        'scope': 'model-specific deterministic inference; replayed only when enabled'}, indent=2))
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_after
print('Pinned deterministic HIP MM/BMM patch verified')
