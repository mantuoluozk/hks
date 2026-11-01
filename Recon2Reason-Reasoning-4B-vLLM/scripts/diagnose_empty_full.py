"""Reconstruct seeded requests; capture 1024 backend token IDs for both flagged inputs."""
import json,random,time,urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
import sglang.benchmark.datasets.image as im
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/empty-output-diagnosis'
OUT.mkdir(exist_ok=True)
base=next(json.loads(x) for x in (ROOT/'results/formal-20261031T163337Z/process1/round2.jsonl').read_text().splitlines() if json.loads(x)['random_input_len']==16384 and json.loads(x)['max_concurrency']==64)
original=im.create_mm_data_row
counter=0
def selected_only(text,images,encoded,output,processor,backend):
 global counter
 index=counter;counter+=1
 if index not in (70,71,98):
  return im.DatasetRow(prompt='',prompt_len=base['input_lens'][index],output_len=output,text_prompt_len=0,vision_prompt_len=0,image_data=[])
 prng=random.getstate();nrng=np.random.get_state()
 row=original(text,images,encoded,output,processor,backend)
 after=np.random.get_state()
 assert random.getstate()==prng and after[0]==nrng[0] and np.array_equal(after[1],nrng[1]) and after[2:]==nrng[2:]
 assert row.prompt_len==base['input_lens'][index]
 return row
# Only skip unrelated rows' CPU token accounting. Sampling and all 128 image RNG
# draws are upstream code; selected rows use the original processor and counts.
im.create_mm_data_row=selected_only
data=im.ImageDataset(num_requests=128,image_count=1,input_len=16384,output_len=1024,range_ratio=1,
 image_content='random',image_format='jpeg',image_resolution='1080p',backend='vllm-chat',random_image_count=False,seed=1).load(model_id='recon2reason-4b')
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
records=[]
for index in (71,98):
 row=data[index]
 content=[{'type':'image_url','image_url':{'url':v}} for v in row.image_data]+[{'type':'text','text':row.prompt}]
 pair=[]
 for skip in (False,True):
  body={'model':'recon2reason-4b','messages':[{'role':'user','content':content}],
   'max_completion_tokens':1024,'temperature':0,'ignore_eos':True,'stream':False,
   'skip_special_tokens':skip,'return_token_ids':True}
  folder=OUT/'requests';folder.mkdir(exist_ok=True)
  (folder/f'full-index{index}-skip{skip}.json').write_text(json.dumps(body))
  req=urllib.request.Request('http://127.0.0.1:30123/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
  t=time.monotonic()
  with op.open(req,timeout=600) as response:r=json.load(response)
  (OUT/f'full-index{index}-skip{skip}.response.json').write_text(json.dumps(r,indent=2))
  ch=r['choices'][0];ids=ch['token_ids']
  pair.append(ids)
  record={'index':index,'skip_special_tokens':skip,'seconds':time.monotonic()-t,'input_tokens':r['usage']['prompt_tokens'],
    'output_tokens':r['usage']['completion_tokens'],'token_counts':dict(Counter(ids)),
    'content_characters':len(ch['message'].get('content') or ''),'content_prefix':(ch['message'].get('content') or '')[:160],
    'finish_reason':ch['finish_reason']}
  records.append(record)
  (OUT/'full-summary.json').write_text(json.dumps(records,indent=2))
  print(json.dumps(record),flush=True)
 print('same token IDs under both rendering flags:',index,pair[0]==pair[1],flush=True)
