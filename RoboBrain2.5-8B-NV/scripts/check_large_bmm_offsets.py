"""Regression: small inputs, BMM output crossing the signed int32 index limit."""
import json
import os
from pathlib import Path
import time
import torch
from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

assert torch.cuda.mem_get_info()[0] > 12 * 1024**3, 'Run with the model service stopped'
batch, m, k, n = 129, 4096, 1, 4096
scales = torch.arange(1, batch + 1, device='cuda', dtype=torch.bfloat16)
a = scales[:, None, None].expand(batch, m, k).contiguous()
b = torch.ones((batch, k, n), device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
start = time.monotonic()
out = bmm_batch_invariant(a, b)
torch.cuda.synchronize()
seconds = time.monotonic() - start
assert out.numel() > 2**31 and a.numel() < 2**31 and b.numel() < 2**31
lo, hi = out.flatten(1).amin(1), out.flatten(1).amax(1)
assert torch.equal(lo, scales) and torch.equal(hi, scales), 'Incorrect values or unwritten output'
record = {'shape': [batch, m, k, n], 'output_elements': out.numel(),
          'last_batch_offset_elements': (batch-1)*m*n, 'dtype':'bfloat16',
          'all_elements_checked_by_per_batch_min_max':True, 'pass':True,
          'kernel_seconds':seconds, 'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
dest = Path(os.environ['RUN_RESULTS_DIR'])/'large-bmm-offset-check.json'
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(record, indent=2))
print(json.dumps(record),flush=True)
