"""Serial HTTP latency self-test; does not claim an official quality comparison."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import requests
from PIL import Image,ImageStat

p=argparse.ArgumentParser()
p.add_argument('--base-url',default='http://127.0.0.1:30003')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--sizes',nargs='+',type=int,default=[512,1024,2048])
a=p.parse_args();assert a.repeats>=2
a.output.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.trust_env=False
response=s.get(a.base_url+'/v1/models',timeout=10);response.raise_for_status()
model=response.json()['data'][0]['id']
prompt='A red ceramic teapot on a wooden table beside a blue cup, soft daylight, clean studio photograph.'
rows=[];summary=[]
for size in a.sizes:
    for i in range(a.repeats+1):
        payload={'model':model,'prompt':prompt,'size':f'{size}x{size}','n':1,
                 'response_format':'b64_json','seed':42,'generator_device':'cpu',
                 'num_inference_steps':8,'true_cfg_scale':1.0,'output_format':'png'}
        t=time.perf_counter();r=s.post(a.base_url+'/v1/images/generations',json=payload,timeout=1800)
        elapsed=time.perf_counter()-t;r.raise_for_status()
        body=r.json();raw=base64.b64decode(body['data'][0].pop('b64_json'))
        image=Image.open(io.BytesIO(raw));image.load()
        assert image.size==(size,size) and max(ImageStat.Stat(image.convert('RGB')).stddev)>1
        row={'size':size,'iteration':i,'warmup':i==0,'request_seconds':elapsed,
             'image_sha256':hashlib.sha256(raw).hexdigest(),'response_metadata':body}
        rows.append(row)
        with (a.output/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    measured=[r['request_seconds'] for r in rows if r['size']==size and not r['warmup']]
    summary.append({'size':size,'measured_requests':len(measured),'warmup_requests':1,
                    'mean_seconds':statistics.mean(measured),'median_seconds':statistics.median(measured),
                    'min_seconds':min(measured),'max_seconds':max(measured),
                    'images_per_second':1/statistics.mean(measured)})
    (a.output/'summary.json').write_text(json.dumps({'protocol':'serial HTTP, PNG base64 response, one excluded warmup per size',
        'prompt':prompt,'seed':42,'generator_device':'cpu','native_steps':8,'n':1,'concurrency':1,
        'cells':summary,'official_accuracy_pass':None,'official_performance_protocol':False},indent=2))
