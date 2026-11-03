"""Online functional checks with original requests, replies, and SSE chunks."""
import argparse
import base64
import concurrent.futures
import io
import json
import os
from pathlib import Path
import re
import time

from PIL import Image,ImageDraw,ImageFont
import requests

MODEL=os.environ['SERVED_NAME'];URL='http://127.0.0.1:'+os.environ['PORT']
def chat(messages,thinking=False,stream=False,extra=None):
    payload={'model':MODEL,'messages':messages,'temperature':0,'top_p':1,'seed':42,
             'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':thinking},'stream':stream}
    if stream:payload['stream_options']={'include_usage':True}
    if extra:payload.update(extra)
    record={'request':payload};start=time.perf_counter()
    with requests.Session() as s:
        s.trust_env=False
        with s.post(URL+'/v1/chat/completions',json=payload,stream=stream,timeout=(10,300)) as res:
            record['http_status']=res.status_code
            if res.status_code!=200:record['error_body']=res.text;return record
            if stream:
                chunks=[];text='';reasoning='';done=False;usage=None;finish=None
                for line in res.iter_lines(chunk_size=1):
                    if not line.startswith(b'data: '):continue
                    if line[6:]==b'[DONE]':done=True;break
                    x=json.loads(line[6:]);chunks.append(x)
                    if x.get('usage'):usage=x['usage']
                    for c in x.get('choices',[]):
                        delta=c.get('delta',{});text+=delta.get('content') or ''
                        reasoning+=delta.get('reasoning') or delta.get('reasoning_content') or ''
                        finish=c.get('finish_reason') or finish
                record.update(chunks=chunks,output=text,reasoning=reasoning,usage=usage,finish_reason=finish,done=done)
            else:
                raw=res.json();m=raw['choices'][0]['message']
                record.update(raw_response=raw,output=m.get('content') or '',reasoning=m.get('reasoning') or m.get('reasoning_content') or '',
                              tool_calls=m.get('tool_calls') or [],usage=raw.get('usage'),finish_reason=raw['choices'][0]['finish_reason'])
    record['seconds']=time.perf_counter()-start
    return record

def user(text):return [{'role':'user','content':text}]
def image(color,text=None):
    img=Image.new('RGB',(512,256),color)
    if text:ImageDraw.Draw(img).text((40,80),text,fill='black',font=ImageFont.load_default(size=48))
    b=io.BytesIO();img.save(b,format='PNG');return b.getvalue()
def part(blob):return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(blob).decode()}}
def images(question,blobs):return [{'role':'user','content':[part(b) for b in blobs]+[{'type':'text','text':question}]}]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    def check(name,messages,predicate,**kwargs):
        try:
            r=chat(messages,**kwargs);r['passed']=r.get('http_status')==200 and bool(predicate(r))
        except Exception as e:r={'passed':False,'error':repr(e),'messages':messages}
        r['id']=name;rows.append(r);(args.output/(name+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:r.get(k) for k in ['id','passed','output','tool_calls','error']},ensure_ascii=False),flush=True)
        return r
    base=user('What is 2 + 2? Answer with the number only.')
    first=check('arithmetic',base,lambda r:r['output'].strip()=='4' and not r['reasoning'] and r['finish_reason']=='stop')
    check('chinese',user('中国的首都是哪里？只回答城市名。'),lambda r:'北京' in r['output'] and not r['reasoning'])
    check('multi-turn',[{'role':'user','content':'Remember the code: HKS2026.'},{'role':'assistant','content':'I will remember HKS2026.'},{'role':'user','content':'What is the code? Output only the code.'}],lambda r:'HKS2026' in r['output'])
    check('thinking',user('What is 15% of 240?'),lambda r:bool(r['reasoning'].strip()) and bool(re.search(r'\b36\b',r['output'])) and r['finish_reason']=='stop',thinking=True)
    check('repeat',base,lambda r:first['passed'] and r['output']==first['output'])
    check('stream',base,lambda r:r['done'] and r['output']==first['output'] and not r['reasoning'] and r['finish_reason']=='stop',stream=True)
    check('thinking-stream',user('What is 12 squared?'),lambda r:r['done'] and bool(r['reasoning'].strip()) and bool(re.search(r'\b144\b',r['output'])),thinking=True,stream=True)
    red=image('red');blue=image('blue');ocr=image('white','HKS2026')
    for n,b in [('red',red),('blue',blue),('ocr',ocr)]: (args.output/(n+'.png')).write_bytes(b)
    check('image-red',images('What color fills this image? Answer with one English word.',[red]),lambda r:'red' in r['output'].lower())
    check('image-blue-zh',images('这张图片主要是什么颜色？',[blue]),lambda r:'蓝' in r['output'] or 'blue' in r['output'].lower())
    check('image-ocr',images('Read the text. Output only the text.',[ocr]),lambda r:'HKS2026' in re.sub(r'\s+','',r['output']))
    check('two-images',images('Name the first color and then the second color in English.',[red,blue]),lambda r:bool(re.search('red.*blue',r['output'].lower(),re.S)))
    tool={'type':'function','function':{'name':'add_numbers','description':'Add two integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}
    prompt=user('Use add_numbers to add 2 and 3. Do not calculate it yourself.')
    def valid_tool(r):
        calls=r['tool_calls']
        if len(calls)!=1 or calls[0]['function']['name']!='add_numbers':return False
        a=json.loads(calls[0]['function']['arguments'])
        return set(a)=={'a','b'} and type(a['a']) is int and type(a['b']) is int and a['a']==2 and a['b']==3
    t=check('tool-call',prompt,valid_tool,extra={'tools':[tool],'tool_choice':'auto'})
    if t.get('tool_calls'):
        msg={'role':'assistant','content':t['output'] or None,'tool_calls':t['tool_calls']}
        returned=prompt+[msg,{'role':'tool','tool_call_id':t['tool_calls'][0]['id'],'content':'5'}]
        check('tool-result',returned,lambda r:bool(re.search(r'\b5\b',r['output'])) and not r['tool_calls'],extra={'tools':[tool],'tool_choice':'none'})
    else:
        fail={'id':'tool-result','passed':False,'error':'No tool call to return result for'};rows.append(fail);(args.output/'tool-result.json').write_text(json.dumps(fail))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results=list(pool.map(lambda _:chat(base),range(4)))
    for i,r in enumerate(results):
        r.update(id=f'concurrent-{i}',passed=r.get('output','').strip()=='4' and not r.get('reasoning'))
        rows.append(r);(args.output/(r['id']+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'passed':all(r['passed'] for r in rows),'total':len(rows),'successful':sum(r['passed'] for r in rows),'scope':'Functional only; synthetic tool exchange is not an Agent benchmark','official_accuracy_pass':None}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    if not summary['passed']:raise SystemExit(1)
if __name__=='__main__':main()
