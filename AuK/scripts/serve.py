"""Set explicit Torch policy in the server and spawn-imported stage process."""
import os
import torch

backend = os.environ.get('AUK_ATTENTION_BACKEND', 'default')
if backend not in {'default', 'math'}:
    raise ValueError(f'Unsupported attention backend: {backend}')
if backend == 'math':
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
print(f'AUK_TORCH_POLICY backend={backend} deterministic={torch.are_deterministic_algorithms_enabled()}', flush=True)
import sglang_omni
print(f'AUK_OMNI_SOURCE={sglang_omni.__file__}', flush=True)

if __name__ == '__main__':
    from sglang_omni.cli.__main__ import app
    app()
