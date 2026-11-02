import importlib.metadata as md
import json,platform,os
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
data={'python':platform.python_version(),'os':platform.platform(),'cpu_threads':os.cpu_count(),'image_page':'https://www.scnet.cn/ui/aihub/image/zack233/2116559999150473218','packages':{}}
for name in ['torch','vllm','vllm-hcu','sglang','triton','transformers','openai','Pillow','tokenizers']:
 try:data['packages'][name]=md.version(name)
 except md.PackageNotFoundError:data['packages'][name]=None
data['torch_hip']=torch.version.hip
data['device']=str(torch.cuda.get_device_properties(0))
data['image_repository']='appstore.scnet.cn:5000/aihub/dcu/zack233/recon2reason-vllm:v1.0.0'
data['image_version']='v1.0.0'
data['platform_allocation']={'accelerator':torch.cuda.get_device_name(0),'cards':torch.cuda.device_count(),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'evidence':'Current runtime device properties and cgroup limits'}
data['cgroup_limits']={}
for name in ['cpu.max','memory.max']:
 p=Path('/sys/fs/cgroup')/name
 if p.exists():data['cgroup_limits'][name]=p.read_text().strip()
(root/'results/environment.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
