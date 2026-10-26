"""Functional checks, deliberately separate from benchmark accuracy."""
import concurrent.futures,datetime,json,os,re,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('SMOKE_OUTPUT_DIR',str(ROOT/'results'/('smoke-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))))
OUT.mkdir(parents=True)
BASE=f"http://{os.environ.get('HOST','127.0.0.1')}:{os.environ.get('PORT','30124')}"
MODEL=os.environ.get('SERVED_NAME','mimo-v2.6-9b')
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(name,messages,thinking=False,stream=False,expected=None):
    body={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':2048,
          'chat_template_kwargs':{'enable_thinking':thinking},'stream':stream}
    started=time.monotonic();first=None;done=False;events=[];answer='';reasoning='';result={}
    try:
        req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(req,timeout=300) as response:
            if stream:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':done=True;break
                    event=json.loads(data);events.append(event)
                    for choice in event.get('choices',[]):
                        delta=choice.get('delta',{})
                        text=delta.get('content') or '';thought=delta.get('reasoning_content') or ''
                        if (text or thought) and first is None:first=time.monotonic()-started
                        answer+=text;reasoning+=thought
                finish=next((c['finish_reason'] for e in reversed(events) for c in e.get('choices',[]) if c.get('finish_reason')),None)
                result={'events':events,'done':done,'finish_reason':finish}
            else:
                result=json.load(response);choice=result['choices'][0];msg=choice['message']
                answer=msg.get('content') or '';reasoning=msg.get('reasoning_content') or ''
                finish=choice['finish_reason']
        checks={'nonempty_content':bool(answer.strip()),'finished':finish=='stop',
                'expected_answer':expected is None or bool(re.search(expected,answer)),
                'stream_complete':not stream or done,
                'thinking_channel':bool(reasoning.strip()) if thinking else not reasoning.strip()}
        record={'name':name,'request':body,'response':result,'answer':answer,'reasoning':reasoning,
                'elapsed_s':time.monotonic()-started,'first_token_s':first,'checks':checks,'pass':all(checks.values())}
    except Exception as exc:
        record={'name':name,'request':body,'pass':False,'error':repr(exc),'elapsed_s':time.monotonic()-started}
    (OUT/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,record['pass'],round(record['elapsed_s'],2),record.get('answer','')[:100],flush=True)
    return record
def user(text):return [{'role':'user','content':text}]
checks=[]
for name,path in [('health','/health'),('models','/v1/models')]:
    started=time.monotonic()
    try:
        with OPENER.open(BASE+path,timeout=15) as response:
            raw=response.read().decode();status=response.status
        passed=status==200
        if name=='models':passed=passed and any(i['id']==MODEL for i in json.loads(raw)['data'])
        record={'name':name,'pass':passed,'response':raw,'status':status,'elapsed_s':time.monotonic()-started}
    except Exception as exc:record={'name':name,'pass':False,'error':repr(exc),'elapsed_s':time.monotonic()-started}
    (OUT/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');checks.append(record)
checks.extend([call('simple',user('只输出数字：2+3等于多少？'),expected=r'\b5\b'),
        call('multiturn',[{'role':'user','content':'记住暗号是青竹。'},{'role':'assistant','content':'好的，暗号是青竹。'},
                          {'role':'user','content':'刚才的暗号是什么？只输出暗号。'}],expected='青竹'),
        call('thinking',user('What is 15% of 240?'),thinking=True,expected=r'\b36\b'),
        call('stream',user('只输出数字：8乘以7是多少？'),stream=True,expected=r'\b56\b'),
        call('thinking-stream',user('What is 12 squared?'),thinking=True,stream=True,expected=r'\b144\b')])
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    checks.extend(pool.map(lambda i:call(f'parallel-{i}',user(f'只输出数字：{i}+10是多少？'),expected=rf'\b{i+10}\b'),range(4)))
summary={'scope':'functional only; not official accuracy evaluation','total':len(checks),'passed':sum(c['pass'] for c in checks),
         'cases':[{'name':c['name'],'pass':c['pass'],'elapsed_s':c['elapsed_s'],'checks':c.get('checks'),'error':c.get('error')} for c in checks]}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
raise SystemExit(0 if summary['passed']==summary['total'] else 1)
