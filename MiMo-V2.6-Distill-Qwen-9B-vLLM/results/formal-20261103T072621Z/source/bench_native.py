"""Run native-token text loads against vLLM completions; retain raw request records.

Dataset tokenization is selected locally because the stock CLI only exposes the
tokenize-prompt flag for SGLang. No chat template or images are added to T01 loads.
The HTTP implementation, timing, scheduling and statistics remain in the client.
"""
import atexit
import dataclasses
import json
import inspect
import os
from pathlib import Path

from sglang.benchmark import serving

OUT=Path(os.environ['BENCH_INPUTS_DIR']);OUT.mkdir(parents=True,exist_ok=False)
records=[]
dataset_fn=serving.get_dataset
request_fn=serving.ASYNC_REQUEST_FUNCS['vllm']
original_source=inspect.getsource(request_fn)
reviewed=Path(__file__).resolve().parents[1]/'sources/benchmark_openai_completions.original.py'
assert original_source.strip()==reviewed.read_text().strip(), 'Benchmark client source changed'
condition='                            if data["choices"][0]["text"]:'
update='''                                output_len = (data.get("usage") or {}).get(
                                    "completion_tokens", output_len
                                )'''
assert original_source.count(condition)==original_source.count(update)==1
corrected=original_source.replace(update,'').replace(condition,
    '                            output_len = (data.get("usage") or {}).get("completion_tokens", output_len)\n'
    '                            if data.get("choices") and data["choices"][0]["text"]:')
assert corrected.count('output_len = request_func_input.output_len')==1
corrected=corrected.replace('output_len = request_func_input.output_len','output_len = 0')
namespace={}
exec(compile(corrected,str(reviewed.with_name('benchmark_openai_completions.patched.py')),'exec'),serving.__dict__,namespace)
request_fn=namespace['async_request_openai_completions']
(OUT/'client-patch.json').write_text(json.dumps({'scope':'Process-local OpenAI completions SSE client',
    'empty_choices_guard':True,'usage_processed_independently_of_text':True,'output_budget_fallback_removed':True},indent=2))

def number(value):
    if hasattr(value,'item'):return value.item()
    raise TypeError(type(value).__name__)

def dataset(args,tokenizer,model_id=None):
    previous=args.tokenize_prompt
    args.tokenize_prompt=True
    try:rows=dataset_fn(args,tokenizer,model_id)
    finally:args.tokenize_prompt=previous
    capacity=json.loads(Path(os.environ['MODEL_PATH'],'config.json').read_text())['text_config']['vocab_size']
    for row in rows:
        assert isinstance(row.prompt,list) and len(row.prompt)==int(row.prompt_len)==args.random_input_len
        assert all(isinstance(i,int) and 0<=i<capacity for i in row.prompt)
    (OUT/'inputs.json').write_text(json.dumps([dataclasses.asdict(r) for r in rows],default=number))
    return rows

async def request(request_func_input,pbar=None):
    response=await request_fn(request_func_input,pbar)
    if response.success and response.output_len<=0:
        response.success=False
        response.error='Missing actual usage.completion_tokens'
    # Keep in memory until benchmark termination so file I/O is outside timing.
    records.append({'request':dataclasses.asdict(request_func_input),'response':dataclasses.asdict(response)})
    return response

def save():
    with (OUT/'requests-and-timing.jsonl').open('w') as f:
        for row in records:f.write(json.dumps(row,default=number)+'\n')

serving.get_dataset=dataset
serving.ASYNC_REQUEST_FUNCS['vllm']=request
atexit.register(save)
if __name__=='__main__':serving.cli_main()
