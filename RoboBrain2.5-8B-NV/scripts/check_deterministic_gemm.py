import json
import os
from pathlib import Path
import torch
from sglang.srt.batch_invariant_ops.batch_invariant_ops import _matmul_persistent_triton, bmm_batch_invariant
torch.manual_seed(20261024)
a = torch.randn(129, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 6144, dtype=torch.bfloat16, device='cuda')
outputs = [_matmul_persistent_triton(a[:n], b) for n in [1, 17, 129]]
torch.cuda.synchronize()
assert all(torch.equal(outputs[0][0], o[0]) for o in outputs)
reference = a.float() @ b.float()
error = (outputs[-1].float() - reference).norm() / reference.norm()
assert torch.isfinite(error) and error < 0.005, error
record = {'batch_rows': [1, 17, 129], 'K':4096, 'N':6144, 'dtype':'bfloat16',
          'first_row_byte_exact_across_batch_sizes': True, 'relative_l2_error_vs_fp32': error.item(),
          'shared_memory_per_block': torch.cuda.get_device_properties(0).shared_memory_per_block}
root = Path(__file__).resolve().parents[1]
bmm_checks = []
for dtype in [torch.bfloat16, torch.float32]:
    x = torch.randn(16, 33, 72, dtype=dtype, device='cuda')
    y = torch.randn(16, 72, 65, dtype=dtype, device='cuda')
    got = bmm_batch_invariant(x, y)
    ref = torch.bmm(x.float(), y.float())
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < (0.005 if dtype == torch.bfloat16 else 0.00001), rel
    single = bmm_batch_invariant(x[:1], y[:1])
    assert torch.equal(single[0], got[0])
    bmm_checks.append({'dtype': str(dtype), 'relative_l2_error_vs_fp32': rel, 'first_batch_exact_match': True})
record['bmm_checks'] = bmm_checks
(Path(os.environ.get('RUN_RESULTS_DIR',str(root/'results'))) / 'deterministic-gemm-check.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
