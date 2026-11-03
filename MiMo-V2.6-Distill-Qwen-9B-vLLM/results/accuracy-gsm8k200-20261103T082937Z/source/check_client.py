"""Exercise the real patched async SSE function with bounded in-memory responses."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import bench_native as client
from sglang.benchmark import serving

serving.args=SimpleNamespace(disable_stream=False,return_logprob=False,top_logprobs_num=0,
                            disable_ignore_eos=False,cache_report=False)
chunks=[];status=200
class Response:
    reason='Synthetic failure'
    @property
    def status(self):return status
    @property
    def content(self):
        async def generate():
            for chunk in chunks:
                await asyncio.sleep(.001)
                yield b'data: '+(chunk.encode() if isinstance(chunk,str) else json.dumps(chunk).encode())+b'\n\n'
        return generate()
    async def text(self):return 'synthetic HTTP error'
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
class Session:
    def post(self,**kwargs):return Response()
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
serving._create_bench_client_session=lambda:Session()
request=serving.RequestFuncInput(prompt=[1,2],api_url='http://local-test/v1/completions',prompt_len=2,output_len=128,model='test',lora_name=None,image_data=None,extra_request_body={})
async def main():
    global chunks,status
    chunks=[{'choices':[{'text':'hello'}]},{'choices':[{'text':' world'}]},
            {'choices':[],'usage':{'completion_tokens':33}},'[DONE]']
    good=await client.request(request)
    assert good.success and good.output_len==33 and good.generated_text=='hello world' and good.ttft>0
    chunks=[{'choices':[{'text':'hello'}]},'[DONE]']
    missing=await client.request(request)
    assert not missing.success and missing.output_len==0 and missing.error=='Missing actual usage.completion_tokens'
    status=500
    failed=await client.request(request)
    assert not failed.success and 'synthetic HTTP error' in failed.error
    result={'passed':True,'usage_only_empty_choices':True,'actual_usage_not_budget':True,'missing_usage_fails':True,'http_failure_preserved':True,'model_requests_sent':0}
    (Path(os.environ['BENCH_INPUTS_DIR'])/'check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
asyncio.run(main())
