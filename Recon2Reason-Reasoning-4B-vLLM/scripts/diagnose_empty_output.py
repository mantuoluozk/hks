"""Replay fixed synthetic requests and compare detokenization without changing sampling."""
import json,time,urllib.request,dataclasses
from collections import Counter
from pathlib import Path
from sglang.benchmark.datasets.image import ImageDataset
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/empty-output-diagnosis'
OUT.mkdir(exist_ok=True)
requests=OUT/'requests'
requests.mkdir(exist_ok=True)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(180):
 try:
  with op.open('http://127.0.0.1:30123/health',timeout=3) as r:
   if r.status==200:break
 except Exception:pass
 time.sleep(5)
else:raise TimeoutError('Readiness timeout')
dataset=ImageDataset(num_requests=128,image_count=1,input_len=16384,output_len=1024,
 range_ratio=1,image_content='random',image_format='jpeg',image_resolution='1080p',
 backend='vllm-chat',random_image_count=False,seed=1).load(model_id='recon2reason-4b')
baseline=[json.loads(x) for x in (ROOT/'results/formal-20261031T163337Z/process1/round2.jsonl').read_text().splitlines()]
baseline=next(x for x in baseline if x['random_input_len']==16384 and x['max_concurrency']==64)
summary=[]
for index in (70,71,98):
 row=dataset[index]
 assert row.prompt_len==baseline['input_lens'][index],(index,row.prompt_len,baseline['input_lens'][index])
 messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':url}} for url in row.image_data]+[{'type':'text','text':row.prompt}]}]
 ids=[]
 for skip in (True,False):
  body={'model':'recon2reason-4b','messages':messages,'max_completion_tokens':64,
   'temperature':0,'ignore_eos':True,'stream':False,'return_token_ids':True,'skip_special_tokens':skip}
  tag=f'index{index}-skip{skip}'
  (requests/f'{tag}.json').write_text(json.dumps(body))
  q=urllib.request.Request('http://127.0.0.1:30123/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
  t=time.monotonic()
  with op.open(q,timeout=600) as r:response=json.load(r)
  (OUT/f'{tag}.response.json').write_text(json.dumps(response,indent=2))
  choice=response['choices'][0]
  token_ids=choice.get('token_ids')
  if token_ids is None:token_ids=choice.get('message',{}).get('token_ids')
  assert token_ids is not None,choice.keys()
  ids.append(token_ids)
  summary.append({'index':index,'skip_special_tokens':skip,'seconds':time.monotonic()-t,
   'content':choice['message'].get('content'),'reasoning':choice['message'].get('reasoning_content'),
   'token_counts':dict(Counter(token_ids)),'usage':response['usage'],'finish_reason':choice['finish_reason']})
 assert ids[0]==ids[1],index
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary[-2:],ensure_ascii=False),flush=True)
