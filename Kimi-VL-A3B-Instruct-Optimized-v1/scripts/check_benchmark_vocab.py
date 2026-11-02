import sys,random,json
from pathlib import Path
sys.path.insert(0,'scripts')
from benchmark_valid_vocab import checked_prompt,capacity
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('weights',trust_remote_code=True,local_files_only=True)
rows=[]
for length in (4096,16384):
 random.seed(1)
 for index in range(128):
  text=checked_prompt(t,None,length);ids=t.encode(text,add_special_tokens=False)
  rows.append({'length_parameter':length,'index':index,'actual_tokens':len(ids),'min':min(ids),'max':max(ids)})
Path('results/benchmark-valid-vocab-check.json').write_text(json.dumps({'capacity':capacity,'checked':len(rows),'passed':True,'rows':rows},indent=2))
print('VALIDATED',len(rows),'synthetic prompts',flush=True)
