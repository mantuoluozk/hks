"""Inspect the pinned Trident reference and native pooling availability."""
import importlib.metadata as md
import hashlib
import json
import os
from pathlib import Path
import torch
import qwen_vl_utils
from embedding_backends.trident_qwen3vl import TridentQwen3VLBackend
from vllm import LLM
import vllm

p=Path(os.environ['MODEL_PATH'])
config=json.loads((p/'config.json').read_text())
embedding=json.loads((p/'mmemb_model.json').read_text())
assert config['architectures']==['Qwen3VLForEmbedding']
assert embedding['pooling']=='last_token' and embedding['normalize'] and embedding['embed_dim'] is None
properties=torch.cuda.get_device_properties(0)
vroot=Path(vllm.__file__).parent
framework=[]
for name in ['model_executor/models/qwen3_vl.py','model_executor/models/qwen3.py','model_executor/models/adapters.py']:
    path=vroot/name
    assert path.is_file(),path
    framework.append({'package':'vllm','path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
result={'model_path':str(p),'packages':{n:md.version(n) for n in ['torch','vllm','transformers','pillow','torchvision']},
        'device':{'name':properties.name,'gcn_arch':properties.gcnArchName,'total_memory':properties.total_memory,
                  'multiprocessors':properties.multi_processor_count,'torch_hip':torch.version.hip},
        'framework_files':framework,
        'qwen_vl_utils_source':qwen_vl_utils.__file__,'reference_backend_import':True,
        'embedding_config':embedding,'embedding_dimension':config['text_config']['hidden_size'],
        'native_route':'Qwen3VLForConditionalGeneration + pooling + convert=embed + LAST token',
        'inference_verified':False,'official_accuracy_pass':None}
out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
