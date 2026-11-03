"""Expose the fixed publisher JSON template to SGLang's tokenizer explicitly."""
import hashlib,json,os
from pathlib import Path
from jinja2 import Environment
ROOT=Path(__file__).resolve().parents[1]
source=Path(os.environ['MODEL_PATH'])/'chat_template.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff'
template=json.loads(source.read_text())['chat_template']
render=Environment().from_string(template).render(messages=[{'role':'user','content':[{'type':'audio_url','audio_url':{'url':'FIXTURE'}}]}],add_generation_prompt=True)
expected='<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n'
assert render==expected,repr(render)
out=ROOT/'runtime';out.mkdir(exist_ok=True);target=out/'qwen3-asr.jinja'
target.write_text(template,encoding='utf-8',newline='\n')
(Path(os.environ['RUN_RESULTS_DIR'])/'template-preparation.json').write_text(json.dumps({'passed':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'template_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'action':'Extract JSON chat_template value unchanged and explicitly pass --chat-template; no weight/tokenizer edits','empty_system_and_audio_render_verified':True},indent=2)+'\n')
print('PUBLISHER_TEMPLATE_READY',flush=True)
