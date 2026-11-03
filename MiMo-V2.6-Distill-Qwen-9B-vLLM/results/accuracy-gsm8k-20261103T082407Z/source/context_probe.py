"""Check 16K token-ID API usage and output capacity outside timed traffic."""
import argparse,json,os,time
from pathlib import Path
import requests

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ids=[1000+i%1000 for i in range(16384)]
payload={'model':os.environ['SERVED_NAME'],'prompt':ids,'temperature':0,'top_p':1,'seed':42,'max_tokens':32,'ignore_eos':True,'stream':False}
with requests.Session() as s:
    s.trust_env=False;start=time.perf_counter();response=s.post('http://127.0.0.1:'+os.environ['PORT']+'/v1/completions',json=payload,timeout=(10,240))
    response.raise_for_status();raw=response.json();elapsed=time.perf_counter()-start
a.output.write_text(json.dumps({'request':payload,'raw_response':raw,'seconds':elapsed},ensure_ascii=False),encoding='utf-8')
assert raw['usage']['prompt_tokens']==16384 and raw['usage']['completion_tokens']==32
assert raw['choices'][0]['text'].strip() and raw['choices'][0]['finish_reason']=='length'
assert 16384+1024<=int(os.environ['MAX_LEN'])
print('NATIVE_CONTEXT_PROBE_PASSED')
