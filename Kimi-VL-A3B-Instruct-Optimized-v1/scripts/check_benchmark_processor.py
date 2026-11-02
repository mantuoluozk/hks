import sys,json,gc
from pathlib import Path
sys.path.insert(0,'scripts')
import benchmark_valid_vocab
from sglang.benchmark.datasets.image import ImageDataset
summary=[]
for length in (4096,16384):
 d=ImageDataset(num_requests=128,image_count=1,input_len=length,output_len=1024,range_ratio=1,image_content='random',image_format='jpeg',image_resolution='1080p',backend='vllm-chat',random_image_count=False,seed=1)
 rows=d.load(model_id='kimi-vl-a3b-instruct')
 assert len(rows)==128
 summary.append({'input_length':length,'processed_requests':len(rows),'passed':True})
 print('PROCESSOR_CHECK',json.dumps(summary[-1]),flush=True)
 del rows;gc.collect()
Path('results/benchmark-processor-check.json').write_text(json.dumps(summary,indent=2))
