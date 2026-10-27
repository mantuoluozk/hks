"""Full BF16 text-to-image path, preserving checkpoint's native eight-step schedule."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from diffusers import QwenImage21Pipeline

out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
pipe=QwenImage21Pipeline.from_pretrained(os.environ['MODEL_PATH'],dtype=torch.bfloat16,local_files_only=True).to('cuda')
initialization=time.perf_counter()-start
steps=[]


def step_end(pipeline,index,timestep,kwargs):
    tensor=kwargs['latents']
    assert torch.isfinite(tensor).all(),index
    steps.append({'step':index,'timestep':float(timestep),'latent_shape':list(tensor.shape)})
    print('completed denoising step',index,flush=True)
    return kwargs


prompt='A red ceramic teapot on a wooden table beside a blue cup, soft daylight, clean studio photograph.'
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize();start=time.perf_counter()
image=pipe(prompt=prompt,width=512,height=512,true_cfg_scale=1.0,use_kv_cache=True,
           generator=torch.Generator('cpu').manual_seed(42),callback_on_step_end=step_end).images[0]
torch.cuda.synchronize();elapsed=time.perf_counter()-start
array=np.asarray(image)
assert image.size==(512,512) and array.std()>1 and len(steps)==8
path=out/'text-to-image-512.png';image.save(path)
result={'prompt':prompt,'seed':42,'generator_device':'cpu','width':512,'height':512,'dtype':'bfloat16','use_kv_cache':True,
        'configured_sample_sigmas':list(pipe.config.sample_sigmas),'actual_scheduler_sigmas':pipe.scheduler.sigmas.float().cpu().tolist(),
        'denoising_steps':steps,'initialization_seconds':initialization,'pipeline_seconds':elapsed,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pixel_standard_deviation':float(array.std()),
        'functional_pass':True,'official_accuracy_pass':None,
        'scope':'512x512 full-pipeline compatibility probe only; not official image quality or final resolution performance.'}
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
