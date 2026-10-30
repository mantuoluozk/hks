"""Synthetic HTTP functionality/performance workloads; no official accuracy claim."""
import argparse, base64, concurrent.futures, io, json, statistics, time, urllib.request, urllib.error
from pathlib import Path
from PIL import Image, ImageDraw

def image_url(size, color):
    im=Image.new('RGB',(size,size),color)
    ImageDraw.Draw(im).rectangle((size//4,size//4,3*size//4,3*size//4),fill='white')
    b=io.BytesIO();im.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def request(base,model,images=(),stream=False,max_tokens=128,messages=None):
    content=[{'type':'text','text':'Describe the image(s) briefly.' if images else 'Reply with one short sentence.'}]+[{'type':'image_url','image_url':{'url':u}} for u in images]
    body={'model':model,'messages':messages or [{'role':'user','content':content}],'temperature':0,'max_tokens':max_tokens,'stream':stream,'chat_template_kwargs':{'enable_thinking':False}}
    started=time.perf_counter()
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r: raw=r.read().decode()
    elapsed=time.perf_counter()-started
    if stream:
        chunks=[json.loads(s[6:]) for s in raw.splitlines() if s.startswith('data: ') and s[6:]!='[DONE]']
        text=''.join(c['choices'][0].get('delta',{}).get('content') or '' for c in chunks if c.get('choices'))
        assert '[DONE]' in raw
        usage=None
    else:
        obj=json.loads(raw);text=obj['choices'][0]['message'].get('content') or '';usage=obj.get('usage')
    assert text.strip(),'empty final content'
    return {'seconds':elapsed,'text':text,'usage':usage,'stream':stream,'image_count':len(images)}

def smoke(base,model,out):
    with urllib.request.urlopen(base+'/health',timeout=10) as r:assert r.status==200
    a=image_url(448,'red');b=image_url(896,'blue');rows=[]
    for name,imgs,stream in [('text',(),False),('single',(a,),False),('large',(b,),False),('two',(a,b),False),('eight',(a,b,a,b,a,b,a,b),False),('stream',(a,),True)]:
        r=request(base,model,imgs,stream);r['case']=name;rows.append(r)
        (out/'smoke.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    follow=request(base,model,messages=[{'role':'user','content':'Remember the word red.'},{'role':'assistant','content':'red'},{'role':'user','content':'Which word did I ask you to remember?'}]);follow['case']='multi_turn';rows.append(follow)
    try:
        request(base,model,max_tokens=-1)
    except urllib.error.HTTPError as e:
        assert e.code==400
        rows.append({'case':'invalid_max_tokens','status':e.code})
    else:raise AssertionError('Invalid max_tokens was accepted')
    (out/'smoke.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    return rows

def bench(base,model,out):
    rows=[]
    for size in (448,896):
        imgs=(image_url(size,'red'),)
        for concurrency in (1,2,4):
            warm=request(base,model,imgs);warm.update(size=size,concurrency=concurrency,warmup=True);rows.append(warm)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                started=time.perf_counter()
                values=list(pool.map(lambda _:request(base,model,imgs),range(3*concurrency)))
                wall=time.perf_counter()-started
            for v in values:v.update(size=size,concurrency=concurrency,warmup=False,batch_wall_seconds=wall);rows.append(v)
            with (out/'requests.jsonl').open('w') as f:
                for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','bench']);p.add_argument('--base',default='http://127.0.0.1:30147');p.add_argument('--model',default='minicpm-v-4.7');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    (smoke if a.mode=='smoke' else bench)(a.base,a.model,a.output)
