"""One screen, six streaming image loads with per-cell warmup and actual usage."""
import base64,concurrent.futures,io,json,os,time,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'results/initial/short';O.mkdir(parents=True)
im=Image.new('RGB',(448,336),'white');ImageDraw.Draw(im).rectangle((40,40,300,270),fill='red');b=io.BytesIO();im.save(b,format='PNG');(O/'input.png').write_bytes(b.getvalue())
url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
body={'model':os.environ['SERVED_NAME'],'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':url}},{'type':'text','text':'Describe this picture in detailed English sentences, including colors, shapes, and their spatial arrangement.'}]}],'temperature':0,'top_p':1,'seed':42,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
def call(budget):
 request={**body,'max_tokens':budget};op=urllib.request.build_opener(urllib.request.ProxyHandler({}));start=time.perf_counter();first=None;content=[];events=[]
 with op.open(urllib.request.Request('http://127.0.0.1:30126/v1/chat/completions',json.dumps(request).encode(),{'Content-Type':'application/json'}),timeout=600) as response:
  for line in response:
   text=line.decode().strip()
   if text=='data: [DONE]':break
   if text.startswith('data: '):
    c=json.loads(text[6:]);events.append(c)
    if c.get('choices'):
     part=c['choices'][0].get('delta',{}).get('content') or ''
     if part and first is None:first=time.perf_counter()-start
     content.append(part)
 elapsed=time.perf_counter()-start;usage=next((c['usage'] for c in reversed(events) if c.get('usage')),None)
 assert usage and usage['completion_tokens']==budget and ''.join(content).strip()
 return {'seconds':elapsed,'ttft_seconds':first,'text':''.join(content),'usage':usage,'events':events}
rows=[]
for concurrency in [1,8,16]:
 for budget in [32,128]:
  warm=call(budget);count=max(4,2*concurrency)
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:results=list(pool.map(lambda _:call(budget),range(count)))
  wall=time.perf_counter()-start
  row={'concurrency':concurrency,'output_budget':budget,'requests':count,'warmup':warm,'responses':results,'wall_seconds':wall,'actual_output_tokens':sum(r['usage']['completion_tokens'] for r in results)}
  row['output_tokens_per_second']=row['actual_output_tokens']/wall;rows.append(row)
  (O/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['warmup','responses']}),flush=True)
(O/'request-template.json').write_text(json.dumps(body,indent=2))
