"""CPU-only verification and preparation of the pinned complete MMLU-Pro dataset."""
import argparse
import hashlib
import json
from pathlib import Path
from mmlu_pro_common import ROOT,SOURCE,prompt,sha,upstream

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,default=ROOT/'data/mmlu-pro')
p.add_argument('--output',type=Path,default=ROOT/'data/mmlu-pro/prepared')
a=p.parse_args()
import pyarrow.parquet as pq
meta=json.loads((ROOT/'sources/mmlu-pro-files.json').read_text())
cfg=json.loads((ROOT/'configs/mmlu-pro.json').read_text())
assert meta['sha']==cfg['dataset_revision']
files={}
splits={}
for split in ['test','validation']:
    filename=f'data/{split}-00000-of-00001.parquet'
    entry=next(e for e in meta['siblings'] if e['rfilename']==filename)
    path=a.data/filename
    assert path.stat().st_size==entry['size'],path
    assert sha(path)==entry['lfs']['sha256'],path
    files[split]={'file':filename,'sha256':sha(path),'bytes':path.stat().st_size}
    splits[split]=pq.read_table(path).to_pylist()
assert len(splits['test'])==cfg['expected_test_rows']
ns,_,_=upstream()
validation=ns['preprocess'](splits['validation'])
assert len(validation)==14 and all(len(v)==5 for v in validation.values())
ids=[r['question_id'] for r in splits['test']]
assert len(ids)==len(set(ids))
assert not ({r['question'].strip() for r in splits['test']}&{r['question'].strip() for r in splits['validation']}), 'Test/validation questions overlap'
a.output.mkdir(parents=True,exist_ok=True)
rows=[]
with (a.output/'test.jsonl').open('w',encoding='utf-8') as out:
    for item in splits['test']:
        options=[opt for opt in item['options'] if opt!='N/A']
        expected=item['answer']
        assert expected in 'ABCDEFGHIJ'[:len(options)]
        assert expected=='ABCDEFGHIJ'[item['answer_index']]
        cooked=dict(item,options=options)
        text=prompt(cooked,validation[item['category']])
        record={'id':item['question_id'],'category':item['category'],'expected':expected,
                'option_count':len(options),'prompt':text,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
        out.write(json.dumps(record,ensure_ascii=False)+'\n');rows.append(record)
manifest={'dataset':meta['id'],'revision':meta['sha'],'test_rows':len(rows),'validation_rows':len(splits['validation']),
          'validation_examples_per_category':5,'categories':sorted(validation),'files':files,
          'prepared_sha256':sha(a.output/'test.jsonl'),'ids':ids,
          'prompt_reference_revision':(SOURCE.parent/'revision.txt').read_text().strip(),
          'prompt_reference_sha256':sha(SOURCE),'inference_executed':False}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'test_rows':len(rows),'validation_rows':len(splits['validation']),
                  'categories':len(validation),'prepared_sha256':manifest['prepared_sha256'],'inference_executed':False}))
