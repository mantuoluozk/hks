"""One warmed pass of the historical 512/128 C1/C4 native-token load."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
summaries=[]
for c in [1,4]:
    env=dict(os.environ,BENCH_INPUTS_DIR=str(a.output/f'c{c}-inputs'))
    command=['python','scripts/bench_native.py','--backend','vllm','--host','127.0.0.1','--port',os.environ['PORT'],
             '--model',os.environ['MODEL_PATH'],'--tokenizer',os.environ['MODEL_PATH'],'--served-model-name',os.environ['SERVED_NAME'],
             '--dataset-name','random-ids','--random-input-len','512','--random-output-len','128','--random-range-ratio','1',
             '--max-concurrency',str(c),'--num-prompts','8','--seed','1','--warmup-requests',str(c),
             '--extra-request-body',json.dumps({'stream_options':{'include_usage':True},'top_p':1,'seed':42}),
             '--output-details','--output-file',str(a.output/'result.jsonl')]
    (a.output/f'c{c}-command.json').write_text(json.dumps(command))
    with (a.output/f'c{c}.log').open('w') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
    row=json.loads((a.output/'result.jsonl').read_text().splitlines()[-1])
    assert row['completed']==8 and row['total_output_tokens']==8*128
    assert len(row['errors'])==8 and not any(row['errors'])
    assert len(row['generated_texts'])==8 and all(t.strip() for t in row['generated_texts'])
    assert row['input_lens']==[512]*8 and row['output_lens']==[128]*8
    summaries.append({k:row[k] for k in ['max_concurrency','completed','duration','output_throughput','mean_ttft_ms','mean_tpot_ms']})
    print(json.dumps(summaries[-1]),flush=True)
(a.output/'summary.json').write_text(json.dumps({'passed':True,'measured_requests':16,'client_warmup_requests':5,'cells':summaries},indent=2))
