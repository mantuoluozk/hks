"""Run native-token text loads against vLLM completions; retain raw request records.

Dataset tokenization is selected locally because the stock CLI only exposes the
tokenize-prompt flag for SGLang. No chat template or images are added to T01 loads.
The HTTP implementation, timing, scheduling and statistics remain in the client.
"""
import atexit
import dataclasses
import json
import os
from pathlib import Path

from sglang.benchmark import serving

OUT=Path(os.environ['BENCH_INPUTS_DIR']);OUT.mkdir(parents=True,exist_ok=False)
records=[]
dataset_fn=serving.get_dataset
request_fn=serving.ASYNC_REQUEST_FUNCS['vllm']

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
