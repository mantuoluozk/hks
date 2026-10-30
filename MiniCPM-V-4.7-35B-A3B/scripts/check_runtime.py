"""Import and config checks only; never load weights."""
import hashlib, importlib.metadata as md, json
from pathlib import Path
import transformers, tokenizers, torch, vllm
from transformers import AutoConfig
from vllm.plugins import load_general_plugins
from minicpm_adapter.model import MiniCPMV4_7ForConditionalGeneration
root=Path(__file__).resolve().parents[1]
load_general_plugins()
assert "MiniCPMV4_7ForConditionalGeneration" in vllm.ModelRegistry.get_supported_archs()
config=AutoConfig.from_pretrained(root/'sources/publisher-configs',local_files_only=True)
assert config.model_type=='minicpmv4_7'
assert config.text_config.model_type=='qwen3_5_moe_text'
assert config.vision_config.num_hidden_layers==27
from vllm.config import ModelConfig
model_config=ModelConfig(model=str(root/'sources/publisher-configs'),tokenizer=str(root/'sources/publisher-configs'),skip_tokenizer_init=True,max_model_len=8192,dtype='bfloat16')
assert model_config.hf_text_config.model_type=='qwen3_5_moe_text'
assert model_config.hf_text_config.num_experts==256
assert transformers.__version__=='5.18.0'
assert tokenizers.__version__=='0.23.1'
result={'passed':True,'weights_loaded':False,'transformers':transformers.__version__,'transformers_path':transformers.__file__,'tokenizers':tokenizers.__version__,'vllm':md.version('vllm'),'torch':torch.__version__,'model_class':MiniCPMV4_7ForConditionalGeneration.__name__,'architecture':config.architectures,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'python/minicpm_adapter').glob('*.py')}}
(root/'results/runtime').mkdir(exist_ok=True)
(root/'results/runtime/runtime-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
