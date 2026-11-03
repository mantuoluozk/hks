"""Read-only pinned-weight, runtime, and native-path checks. No download."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ['RUN_RESULTS_DIR']);OUT.mkdir(parents=True,exist_ok=True)
model=Path(os.environ['MODEL_PATH'])
source=json.loads((model/'upstream-source.json').read_text())
assert source['repo']=='XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B'
assert source['revision']=='2367e865d009c13ac81713a2878291d33ab28177'
items=[f for f in json.loads((ROOT/'sources/weight-files.json').read_text())['Data']['Files'] if f['Name'] not in ['README.md','.gitattributes','configuration.json']]
assert len(items)==15
proof=[]
for f in items:
    p=model/f['Name'];assert p.stat().st_size==f['Size'],f['Name']
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(16*1024**2),b''):h.update(b)
    assert h.hexdigest()==f['Sha256'],f['Name']
    proof.append({'path':f['Name'],'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    print('VERIFIED',f['Name'],flush=True)
(OUT/'weights-check.json').write_text(json.dumps({'revision':source['revision'],'files':proof,'passed':True},indent=2))
import torch
import vllm
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
reasoning=ReasoningParserManager.get_reasoning_parser('mimo')
tools=ToolParserManager.get_tool_parser('mimo')
base=Path(vllm.__file__).parent
paths=['model_executor/models/qwen3_5.py','model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py','model_executor/layers/mamba/mamba_utils.py','reasoning/qwen3_engine_reasoning_parser.py','tool_parsers/qwen3_engine_tool_parser.py']
snap={}
for rel in paths:
    p=base/rel;snap[rel]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,ROOT/'sources'/p.name)
record={'python':sys.version,'packages':{n:metadata.version(n) for n in ['torch','vllm','triton','transformers','tokenizers','sglang']},
        'gpu':torch.cuda.get_device_name(0),'memory':torch.cuda.mem_get_info(),'hip':torch.version.hip,
        'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        'native_model':Qwen3_5ForConditionalGeneration.__name__,'reasoning_parser':reasoning.__name__,'tool_parser':tools.__name__,
        'source_sha256':snap,'model_inference_verified':False}
(OUT/'environment.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
