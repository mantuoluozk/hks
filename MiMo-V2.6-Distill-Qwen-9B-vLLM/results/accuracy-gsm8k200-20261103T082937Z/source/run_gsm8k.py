"""The same 200 GSM8K questions and scoring protocol as the published SGLang run."""
import argparse,concurrent.futures,hashlib,json,os,time,urllib.request,urllib.error
from pathlib import Path
from gsm8k_score import prompt,extract_answer,score,INVALID_ANS,summarize
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
data=ROOT/'data/gsm8k/test.jsonl';items=[json.loads(s) for s in data.read_text(encoding='utf-8').splitlines()];assert len(items)==200
proof=json.loads((data.parent/'proof.json').read_text());assert hashlib.sha256(data.read_bytes()).hexdigest()==proof['prepared_sha256']
requests=ROOT/'data/gsm8k/requests'/a.output.parent.name;requests.mkdir(parents=True)
protocol={'dataset':proof,'few_shot':0,'workers':8,'temperature':0,'top_p':1,'request_seed':None,'server_seed':42,'max_tokens':4096,
          'enable_thinking':True,'ignore_eos':False,'retries':0,'request_timeout_s':600,
          'prompt':'Solve the math problem. Explain briefly, then finish with #### followed by the final numeric answer.\n\n{question}',
          'scorer':json.loads((ROOT/'sources/sglang-quality-reference.json').read_text()),
          'scope':'Same fixed 200 GSM8K sample and prompt/scoring as SGLang SCNet evaluation; not publisher/Qwen3.5-9B matched acceptance'}
(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
base='http://127.0.0.1:'+os.environ['PORT']
def run(task):
    index,item=task;text=prompt(item)
    body={'model':os.environ['SERVED_NAME'],'messages':[{'role':'user','content':text}],
          'temperature':0,'top_p':1,'max_tokens':4096,'chat_template_kwargs':{'enable_thinking':True}}
    blob=json.dumps(body,ensure_ascii=False).encode();path=requests/f'{index:04d}.json';path.write_bytes(blob)
    record={'index':index,'question_sha256':hashlib.sha256(item['question'].encode()).hexdigest(),
            'request_sha256':hashlib.sha256(blob).hexdigest(),'request_path':str(path),
            'expected':item['answer'],'error':None,'truncated':False}
    started=time.monotonic()
    try:
        op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req=urllib.request.Request(base+'/v1/chat/completions',data=blob,headers={'Content-Type':'application/json'})
        with op.open(req,timeout=600) as response:raw=json.load(response)
        choice=raw['choices'][0];predicted,correct=score('gsm8k',choice['message'].get('content') or '',record['expected'])
        record.update(response=raw,predicted=predicted,correct=correct,truncated=choice['finish_reason']=='length')
    except Exception as e:
        record.update(error=repr(e),predicted=INVALID_ANS,correct=False)
        if isinstance(e,urllib.error.HTTPError):record['http_error_body']=e.read().decode(errors='replace')
    record['elapsed_s']=time.monotonic()-started
    return record
rows=[]
with (a.output/'predictions.jsonl').open('w',encoding='utf-8') as output:
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        futures=[pool.submit(run,task) for task in enumerate(items)]
        for future in concurrent.futures.as_completed(futures):
            record=future.result()
            rows.append(record);output.write(json.dumps(record,ensure_ascii=False)+'\n');output.flush()
            if len(rows)%25==0:print('Completed',len(rows),'of 200',flush=True)
summary=summarize(rows);(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
raise SystemExit(0 if summary['technical_checks_passed'] else 1)
