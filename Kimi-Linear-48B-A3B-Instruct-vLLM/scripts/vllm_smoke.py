#!/usr/bin/env python3
"""Six functional groups with raw evidence and a nonzero failure exit status."""
import argparse,json,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:30149')
    p.add_argument('--model',default='kimi-linear-48b-a3b')
    p.add_argument('--timeout',type=float,default=600)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def ask(label,messages,stream=False,max_tokens=128):
        body={'model':a.model,'messages':messages,'temperature':0,'max_tokens':max_tokens,'stream':stream}
        (a.output/f'{label}.request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2),encoding='utf-8')
        t=time.monotonic()
        try:
            req=urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with opener.open(req,timeout=a.timeout) as response:raw=response.read().decode()
            (a.output/f'{label}.response.txt').write_text(raw,encoding='utf-8')
            if stream:
                lines=[line[6:] for line in raw.splitlines() if line.startswith('data: ')]
                if '[DONE]' not in lines:raise ValueError('Stream missing DONE')
                chunks=[json.loads(line) for line in lines if line!='[DONE]']
                text=''.join(c['choices'][0].get('delta',{}).get('content') or '' for c in chunks if c.get('choices'))
            else:text=json.loads(raw)['choices'][0]['message'].get('content') or ''
            if not text.strip() or set(text.strip())=={'!'}:raise ValueError('Empty/degenerate response')
            return {'ok':True,'content':text,'seconds':time.monotonic()-t}
        except urllib.error.HTTPError as e:
            raw=e.read().decode(errors='replace');(a.output/f'{label}.response.txt').write_text(raw,encoding='utf-8')
            return {'ok':False,'status':e.code,'error':raw}
        except Exception as e:return {'ok':False,'error':repr(e)}
    def user(text):return [{'role':'user','content':text}]
    checks=[]
    cases=[('english',user('Reply with the word READY only.'),'READY',False),
           ('chinese',user('计算2+3，只输出数字。'),'5',False),
           ('multi_turn',[{'role':'user','content':'Remember the number 5.'},{'role':'assistant','content':'5'},
              {'role':'user','content':'Add one to that number. Answer with one number only.'}],'6',False),
           ('stream',user('Reply with the word READY only.'),'READY',True)]
    for label,messages,expected,stream in cases:
        result=ask(label,messages,stream);passed=result['ok'] and result['content'].strip()==expected
        checks.append({'case':label,'passed':passed,**result})
    rejected=ask('invalid',user('Hi'),max_tokens=-1)
    checks.append({'case':'invalid_parameter','passed':not rejected['ok'] and rejected.get('status') in (400,422),'response':rejected})
    messages=user('What is 2+3? Answer with one number only.')
    serial=ask('serial',messages)
    with ThreadPoolExecutor(4) as pool:parallel=list(pool.map(lambda i:ask(f'parallel{i}',messages),range(4)))
    passed=all(x['ok'] and x.get('content','').strip()=='5' for x in [serial,*parallel])
    checks.append({'case':'serial_parallel_consistency','passed':passed,'serial':serial,'parallel':parallel})
    summary={'base':a.base,'model':a.model,'checks':checks,'passed':all(x['passed'] for x in checks)}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'checks':len(checks),'passed':summary['passed'],'output':str(a.output)},ensure_ascii=False))
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
