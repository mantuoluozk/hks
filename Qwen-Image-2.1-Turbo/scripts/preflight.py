"""Record the isolated runtime and confirm native DTK torch is retained."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import diffusers
from diffusers import QwenImage21Pipeline
import transformers
import tokenizers
import torch

ROOT=Path(__file__).resolve().parents[1]
assert transformers.__version__=='5.17.0'
assert tokenizers.__version__=='0.23.1'
runtime=Path(os.environ.get('HKS_RUNTIME_ROOT',str(ROOT/'runtime'))).resolve()
assert Path(diffusers.__file__).resolve().is_relative_to(runtime)
assert Path(transformers.__file__).resolve().is_relative_to(runtime)
assert Path(tokenizers.__file__).resolve().is_relative_to(runtime)
launcher_check=None
if os.environ.get('HKS_REQUIRE_BAKED_RUNTIME')=='1':
    assert runtime==Path('/opt/qwen-image21').resolve(),runtime
    installed=runtime/'model/scripts/serve_sglang.py'
    shared=ROOT/'scripts/serve_sglang.py'
    launcher_check={'image_sha256':hashlib.sha256(installed.read_bytes()).hexdigest(),
                    'executed_source_sha256':hashlib.sha256(shared.read_bytes()).hexdigest()}
    assert launcher_check['image_sha256']==launcher_check['executed_source_sha256'],'Image launcher differs from executed source'
assert 'dtk2604' in metadata.version('torch') and not Path(torch.__file__).resolve().is_relative_to((ROOT/'runtime').resolve())
device=torch.cuda.get_device_properties(0)
config=json.loads((Path(os.environ['MODEL_PATH'])/'model_index.json').read_text())
assert config['_class_name']=='QwenImage21Pipeline' and len(config['sample_sigmas'])==8
result={'runtime_root':str(runtime),'baked_runtime':runtime==Path('/opt/qwen-image21').resolve(),
        'launcher_integrity':launcher_check,
        'packages':{m.__name__:{'version':m.__version__,'path':m.__file__} for m in [torch,diffusers,transformers,tokenizers]},
        'torch_distribution_version':metadata.version('torch'),
        'device':{'name':device.name,'gcn_arch':device.gcnArchName,'total_memory':device.total_memory,'compute_units':device.multi_processor_count},
        'pipeline_class':QwenImage21Pipeline.__name__,'checkpoint_config':config,'official_accuracy_pass':None}
out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
