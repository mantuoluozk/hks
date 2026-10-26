"""Local paired quality regression on fixed GSM8K/MMLU samples, not official acceptance."""
import argparse,concurrent.futures,datetime,hashlib,json,os,re,time,urllib.request
from decimal import Decimal,InvalidOperation
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(text):
    try:return Decimal(text.replace(',','').strip())
    except InvalidOperation:return None
def score(dataset,content,truth):
    text=content.replace('**','').strip()
    if dataset=='gsm8k':
        matches=re.findall(r'####\s*(-?\d[\d,]*(?:\.\d+)?)',text)
        prediction=matches[-1] if matches else None
        return prediction,prediction is not None and number(prediction)==number(truth)
    matches=re.findall(r'(?:^|\n)\s*ANSWER:\s*([ABCD])\s*[.。]?\s*$',text,re.I)
    prediction=matches[-1].upper() if matches else None
    return prediction,prediction==truth
def prompt(dataset,item):
    if dataset=='gsm8k':return 'Solve the math problem. Explain briefly, then finish with #### followed by the final numeric answer.\n\n'+item['question']
    choices='\n'.join(f'{letter}. {text}' for letter,text in zip('ABCD',item['choices']))
    return 'Answer this multiple-choice question. Finish with a line exactly like ANSWER: X, replacing X with A, B, C, or D.\n\n'+item['question']+'\n'+choices

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args();assert 1<=args.workers<=64;args.output.mkdir(parents=True)
    runtime=json.loads(Path(os.environ.get('RUN_RESULTS_DIR','results'),'runtime-config.json').read_text())
    protocol={'temperature':0,'top_p':1,'max_tokens':4096,'enable_thinking':True,'few_shot':0,'workers':args.workers,
              'sample_seed':20261023,'scoring':'GSM8K final #### decimal; MMLU final ANSWER: A/B/C/D; strip bold markup; no random fallback',
              'scope':'fixed 200+200 public samples; not official subset or publisher GPU accuracy baseline',
              'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned')}
    tasks=[];sources={}
    for dataset in ['gsm8k','mmlu']:
        path=ROOT/'data'/f'{dataset}.jsonl';items=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len(items)==200,(dataset,len(items))
        sources[dataset]={'prepared_file':str(path.relative_to(ROOT)),'sha256':digest(path),'count':len(items)}
        for index,item in enumerate(items):tasks.append((dataset,index,item))
    manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':protocol,'sources':sources,
              'runtime':runtime,'official_accuracy_pass':None}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    base=f"http://{os.environ.get('HOST','127.0.0.1')}:{os.environ.get('PORT','30124')}"
    def run(task):
        dataset,index,item=task;text=prompt(dataset,item);started=time.monotonic()
        record={'dataset':dataset,'index':index,'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'expected':item['answer']}
        body={'model':os.environ.get('SERVED_NAME','mimo-v2.6-9b'),'messages':[{'role':'user','content':text}],
              'temperature':0,'top_p':1,'max_tokens':4096,'chat_template_kwargs':{'enable_thinking':True}}
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=600) as response:raw=json.load(response)
            choice=raw['choices'][0];content=choice['message'].get('content') or ''
            predicted,correct=score(dataset,content,item['answer'])
            record.update(response=raw,predicted=predicted,correct=correct,truncated=choice['finish_reason']=='length',error=None)
        except Exception as exc:record.update(correct=False,predicted=None,error=repr(exc),truncated=False)
        record['elapsed_s']=time.monotonic()-started
        return record
    records=[]
    with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as output:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures=[pool.submit(run,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                record=future.result();records.append(record);output.write(json.dumps(record,ensure_ascii=False)+'\n');output.flush()
                if len(records)%25==0:print('Completed',len(records),'of',len(tasks),flush=True)
    summary={'scope':protocol['scope'],'official_accuracy_pass':None,'datasets':{}}
    for dataset in sources:
        rows=[r for r in records if r['dataset']==dataset];assert len(rows)==200
        correct=sum(r['correct'] for r in rows)
        summary['datasets'][dataset]={'correct':correct,'total':len(rows),'accuracy_percent':100*correct/len(rows),
            'errors':sum(r['error'] is not None for r in rows),'truncated':sum(r['truncated'] for r in rows),
            'unparsed':sum(r['predicted'] is None for r in rows)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    raise SystemExit(1 if any(v['errors'] for v in summary['datasets'].values()) else 0)
if __name__=='__main__':main()
