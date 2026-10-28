"""Serial end-to-end RGB requests, including vision preprocessing and generation."""
import argparse,base64,hashlib,io,json,statistics,time,urllib.request
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:30006');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];summary=[]
    encoded=[]
    for color in [(230,30,30),(30,30,230)]:
        b=io.BytesIO();Image.new('RGB',(384,384),color).save(b,format='PNG');encoded.append(base64.b64encode(b.getvalue()).decode())
    for count in [1,2,8]:
        measured=[]
        for repeat in range(4):
            payload={'images':[encoded[i%2] for i in range(count)],'prompt':'Describe the visible colors and shapes.',
                     'max_tokens':64,'temperature':0,'seed':42}
            req=urllib.request.Request(a.url+'/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            start=time.perf_counter()
            with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
            seconds=time.perf_counter()-start
            assert response['images']==count and response['output_tokens']>0
            row={'images':count,'repeat':repeat,'warmup':repeat==0,'seconds':seconds,'response':response,
                 'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
            rows.append(row)
            with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if repeat:measured.append(row)
        summary.append({'images':count,'requests':len(measured),'mean_seconds':statistics.mean(x['seconds'] for x in measured),
            'output_tokens':sum(x['response']['output_tokens'] for x in measured),
            'output_tokens_per_second':sum(x['response']['output_tokens'] for x in measured)/sum(x['seconds'] for x in measured)})
    (out/'summary.json').write_text(json.dumps({'concurrency':1,'max_tokens':64,'temperature':0,'image_size':[384,384],
        'includes_visual_frontend':True,'synthetic_workload':True,'results':summary},indent=2)+'\n')

if __name__=='__main__':main()
