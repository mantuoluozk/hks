"""Compare text/vision RoPE config interpretation before and after SGLang imports."""
import json
import os
from pathlib import Path
import runpy
from transformers import AutoConfig,Qwen3VLConfig

ROOT=Path(__file__).resolve().parents[1]


def capture():
    records=[]
    for cls in [AutoConfig,Qwen3VLConfig]:
        config=cls.from_pretrained(str(Path(os.environ['MODEL_PATH'])/'text_encoder'),local_files_only=True)
        records.append({'loader':cls.__name__,'class':type(config).__module__+'.'+type(config).__name__,
                        'vision_class':type(config.vision_config).__module__+'.'+type(config.vision_config).__name__,
                        'vision_rope':{k:v for k,v in vars(config.vision_config).items() if 'rope' in k},
                        'text_rope':{k:v for k,v in vars(config.text_config).items() if 'rope' in k}})
    return records


before=capture()
runpy.run_path(str(ROOT/'scripts/serve_sglang.py'),run_name='hks_config_probe')
after=capture()
out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
(out/'config-comparison.json').write_text(json.dumps({'before':before,'after':after,'configuration_equivalent':before==after},indent=2))
print(json.dumps({'before':before,'after':after},indent=2),flush=True)
assert before==after,'SGLang import changed Qwen3-VL positional configuration'
