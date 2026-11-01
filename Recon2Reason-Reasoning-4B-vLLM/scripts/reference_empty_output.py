"""Publisher Transformers reference for the two exact seeded synthetic inputs."""
import base64,copy,io,json,os,time,signal
from collections import Counter
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/empty-output-diagnosis'
for _ in range(240):
 p=OUT/'full-summary.json'
 if p.exists() and len(json.loads(p.read_text()))==4:break
 time.sleep(5)
else:raise TimeoutError('Waiting for token-ID replay')
pid_file=ROOT/'results/server.pid'
if not pid_file.exists():pid_file=OUT/'server.pid'
pid=int(pid_file.read_text())
cmdline=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'vllm.entrypoints.openai.api_server' in cmdline and b'Recon2Reason-Reasoning-4B' in cmdline
os.kill(pid,signal.SIGTERM)
for _ in range(60):
 if not Path(f'/proc/{pid}').exists():break
 time.sleep(2)
else:
 if Path(f'/proc/{pid}/cmdline').read_bytes()==cmdline:os.killpg(pid,signal.SIGKILL)
import torch,transformers
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
model_path='/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B'
processor=AutoProcessor.from_pretrained(model_path)
model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
results=[]
for index in (71,98):
 body=json.loads((OUT/f'requests/full-index{index}-skipFalse.json').read_text())
 content=body['messages'][0]['content']
 images=[Image.open(io.BytesIO(base64.b64decode(x['image_url']['url'].split(',',1)[1]))).convert('RGB') for x in content if x['type']=='image_url']
 text=next(x['text'] for x in content if x['type']=='text')
 chat=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':text}]}]
 rendered=processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
 inputs=processor(text=[rendered],images=images,return_tensors='pt').to('cuda')
 config=copy.deepcopy(model.generation_config)
 config.do_sample=False;config.eos_token_id=None;config.max_new_tokens=64
 with torch.inference_mode():tokens=model.generate(**inputs,generation_config=config)[0,inputs.input_ids.shape[1]:].tolist()
 vllm=json.loads((OUT/f'full-index{index}-skipFalse.response.json').read_text())
 reference={'index':index,'framework':'Transformers','version':transformers.__version__,'dtype':'bfloat16','attention':'sdpa',
   'input_tokens':inputs.input_ids.shape[1],'new_tokens':tokens,'token_counts':dict(Counter(tokens)),
   'content':processor.tokenizer.decode(tokens,skip_special_tokens=False),
   'same_as_vllm_prefix':tokens==vllm['choices'][0]['token_ids'][:64]}
 results.append(reference)
 (OUT/'publisher-reference.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(reference),flush=True)
 del inputs
