"""Plan by default; --run evaluates a local MiMo service with an auditable denominator."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from mmlu_pro_common import ROOT,SOURCE,aggregate,assert_auk_idle,score_choice,sha,read_jsonl

def http_json(url,body=None,timeout=30):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None
    request=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=timeout) as response:return json.load(response)

def http_health(url):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url,timeout=30) as response:
        if response.status!=200:raise RuntimeError('Model health check failed')

def request_body(cfg,item):
    return {'model':cfg['model'],'messages':[{'role':'user','content':item['prompt']}],
            **{k:cfg[k] for k in ['temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','max_tokens','seed']},
            'stream':False,'chat_template_kwargs':{'enable_thinking':cfg['enable_thinking']}}

def evaluate(cfg,items,out,full,send=http_json,resume=False,provenance=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ids=[r['id'] for r in items]
    fingerprint=hashlib.sha256(json.dumps({'config':cfg,'ids':ids,'prompts':[r['prompt_sha256'] for r in items],
                                         'provenance':provenance},sort_keys=True).encode()).hexdigest()
    manifest_path=out/'manifest.json'
    manifest={'config':cfg,'ids':ids,'full_test_split':full,'fingerprint':fingerprint,
              'provenance':provenance,'created_at':datetime.now(timezone.utc).isoformat(),
              'retry_wrong_answers':False,'parse_reasoning_as_final_answer':False}
    records=[]
    if manifest_path.exists():
        if not resume:raise RuntimeError('Output exists; use --resume or a new output directory')
        assert json.loads(manifest_path.read_text())['fingerprint']==fingerprint,'Resume configuration/protocol mismatch'
        records=read_jsonl(out/'predictions.jsonl')
        aggregate(records,ids,cfg['reference_accuracy_percent'],full)
    else:
        if resume:raise RuntimeError('No existing manifest to resume')
        assert not (out/'predictions.jsonl').exists(),'Unowned predictions file'
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    completed={r['id'] for r in records}
    with (out/'predictions.jsonl').open('a',encoding='utf-8') as log:
        for item in items:
            if item['id'] in completed:continue
            body=request_body(cfg,item)
            result={'id':item['id'],'category':item['category'],'expected':item['expected'],
                    'prompt_sha256':item['prompt_sha256'],'error':None,'correct':False,'predicted':None,
                    'unparsed':True,'truncated':False}
            started=time.perf_counter()
            try:
                raw=send(cfg['base_url'].rstrip('/')+'/v1/chat/completions',body,timeout=cfg['timeout_seconds'])
                result.update(response=raw,**score_choice(raw,item['expected'],item['option_count']))
            except Exception as error:result['error']=type(error).__name__+': '+str(error)
            result['seconds']=time.perf_counter()-started
            log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush();records.append(result)
            summary=aggregate(records,ids,cfg['reference_accuracy_percent'],full)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            if len(records)%25==0 or result['error'] or len(records)==len(items):
                print(json.dumps({k:summary[k] for k in ['completed','expected','accuracy_percent','errors','truncated','unparsed']}),flush=True)
    summary=aggregate(records,ids,cfg['reference_accuracy_percent'],full)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/mmlu-pro.json')
    p.add_argument('--prepared',type=Path,default=ROOT/'data/mmlu-pro/prepared')
    p.add_argument('--output',type=Path)
    p.add_argument('--limit',type=int,help='Pilot only: never reports the full-test numeric threshold')
    p.add_argument('--tokenizer',type=Path,default=os.environ.get('MODEL_PATH'))
    p.add_argument('--run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.run:assert_auk_idle()
    cfg=json.loads(a.config.read_text())
    url=urllib.parse.urlparse(cfg['base_url'])
    assert url.scheme=='http' and url.hostname in {'127.0.0.1','localhost','::1'} and not url.username,'Only local benchmark endpoints are allowed'
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    assert manifest['revision']==cfg['dataset_revision']
    assert sha(a.prepared/'test.jsonl')==manifest['prepared_sha256']
    assert sha(SOURCE)==manifest['prompt_reference_sha256'],'Prompt reference changed after preparation'
    items=read_jsonl(a.prepared/'test.jsonl')
    assert len(items)==cfg['expected_test_rows'] and [r['id'] for r in items]==manifest['ids']
    if a.limit is not None:
        assert 0<a.limit<=len(items)
        items=items[:a.limit]
    full=len(items)==cfg['expected_test_rows']
    plan={'dataset':'MMLU-Pro','selected':len(items),'full_test_split':full,
          'reference_accuracy_percent':cfg['reference_accuracy_percent'],'inference_executed':False,
          'context_length':cfg['context_length'],'max_tokens':cfg['max_tokens'],
          'prompt_protocol':'fixed upstream 5-shot CoT','exact_qwen_benchmark_protocol_confirmed':False}
    tokenizer_info=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
        lengths=[len(tokenizer.apply_chat_template(request_body(cfg,item)['messages'],tokenize=True,
                      add_generation_prompt=True,enable_thinking=cfg['enable_thinking'])) for item in items]
        minimum=max(lengths)+cfg['max_tokens']
        assert minimum<=cfg['context_length'],f'Required context {minimum} exceeds configured context'
        tokenizer_info={'path':str(a.tokenizer),'maximum_prompt_tokens':max(lengths),'minimum_context':minimum,
                        'tokenizer_config_sha256':sha(a.tokenizer/'tokenizer_config.json')}
        plan['tokenizer_preflight']=tokenizer_info
    print(json.dumps(plan,indent=2))
    if not a.run:return
    if not a.output:p.error('--run requires --output')
    if not a.tokenizer:p.error('--run requires a local --tokenizer or MODEL_PATH for the context check')
    if cfg['workers']!=1:raise ValueError('Prepared protocol uses one request at a time')
    base=cfg['base_url'].rstrip('/')
    http_health(base+'/health')
    models=http_json(base+'/v1/models')
    assert cfg['model'] in {r['id'] for r in models['data']},'Unexpected served model'
    model_info=http_json(base+'/model_info')
    raw_server_info=http_json(base+'/server_info')
    server_info={k:raw_server_info.get(k) for k in ['context_length','tp_size','dtype','mem_fraction_static',
                'attention_backend','reasoning_parser','version','max_running_requests']}
    assert model_info.get('reasoning_parser')=='mimo','Expected the validated MiMo reasoning parser'
    actual_context=server_info.get('context_length')
    assert isinstance(actual_context,int) and actual_context>=tokenizer_info['minimum_context'],'Server context is insufficient or could not be verified'
    provenance={'data_manifest':manifest,'tokenizer':tokenizer_info,'server_info':server_info,
                'config_file_sha256':sha(a.config)}
    summary=evaluate(cfg,items,a.output,full,resume=a.resume,provenance=provenance)
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['complete'] and summary['errors']==0 else 2)

if __name__=='__main__':main()
