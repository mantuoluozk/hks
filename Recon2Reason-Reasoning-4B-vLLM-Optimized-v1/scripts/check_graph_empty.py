"""Check this graph service's token IDs for any hidden-only matrix outputs."""
import hashlib,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
job=Path(sys.argv[1]);out=job/'graph-empty-output';out.mkdir()
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
records=[]
reference=json.loads((ROOT/'sources/special-probe-request-hashes.json').read_text(encoding='utf-8'))
generated=None
def request_bytes(index,skip,base):
 global generated
 name=f'full-index{index}-skip{skip}.json'
 historical=ROOT/'results/empty-output-diagnosis/requests'/name
 if historical.exists():body=historical.read_bytes()
 else:
  if generated is None:
   import random,numpy as np
   import sglang.benchmark.datasets.image as im
   original=im.create_mm_data_row;counter=0
   def selected(text,images,encoded,output,processor,backend):
    nonlocal counter
    i=counter;counter+=1
    if i not in (71,98):return im.DatasetRow(prompt='',prompt_len=base['input_lens'][i],output_len=output,text_prompt_len=0,vision_prompt_len=0,image_data=[])
    prng=random.getstate();nrng=np.random.get_state()
    row=original(text,images,encoded,output,processor,backend);after=np.random.get_state()
    assert random.getstate()==prng and after[0]==nrng[0] and np.array_equal(after[1],nrng[1]) and after[2:]==nrng[2:]
    assert row.prompt_len==base['input_lens'][i]
    return row
   im.create_mm_data_row=selected
   try:generated=im.ImageDataset(num_requests=128,image_count=1,input_len=16384,output_len=1024,range_ratio=1,image_content='random',image_format='jpeg',image_resolution='1080p',backend='vllm-chat',random_image_count=False,seed=1).load(model_id='recon2reason-4b')
   finally:im.create_mm_data_row=original
  row=generated[index]
  content=[{'type':'image_url','image_url':{'url':v}} for v in row.image_data]+[{'type':'text','text':row.prompt}]
  body=json.dumps({'model':'recon2reason-4b','messages':[{'role':'user','content':content}],'max_completion_tokens':1024,'temperature':0,'ignore_eos':True,'stream':False,'skip_special_tokens':skip,'return_token_ids':True}).encode()
 assert hashlib.sha256(body).hexdigest()==reference[name]
 return body
for repeat in (1,2):
 for row in map(json.loads,(job/f'process1/round{repeat}.jsonl').read_text(encoding='utf-8').splitlines()):
  for index,text in enumerate(row['generated_texts']):
   if text.strip():continue
   assert (row['random_input_len'],row['max_concurrency'])==(16384,64)
   assert index in (71,98),('Unexpected empty output',index)
   record={'repeat':repeat,'index':index,'input_tokens':row['input_lens'][index],'responses':[]}
   for skip in (False,True):
    body=request_bytes(index,skip,row);data=json.loads(body)
    assert data['skip_special_tokens']==skip and data['return_token_ids'] is True
    target=out/f'round{repeat}-index{index}-skip{skip}'
    target.with_suffix('.request.json').write_bytes(body)
    req=urllib.request.Request('http://127.0.0.1:30123/v1/chat/completions',body,{'Content-Type':'application/json'})
    with op.open(req,timeout=600) as response:raw=response.read()
    target.with_suffix('.response.json').write_bytes(raw)
    r=json.loads(raw);ch=r['choices'][0]
    assert ch['token_ids']==[151673]*1024
    assert r['usage']['prompt_tokens']==row['input_lens'][index]
    assert r['usage']['completion_tokens']==1024 and ch['finish_reason']=='length'
    assert ch['message']['content']==('' if skip else '<|feat_pad|>'*1024)
    record['responses'].append({'path':target.with_suffix('.response.json').name,'sha256':hashlib.sha256(raw).hexdigest(),'request_sha256':hashlib.sha256(body).hexdigest()})
   records.append(record)
(out/'summary.json').write_text(json.dumps(records,indent=2))
print('Current graph-service hidden-special checks:',len(records))
