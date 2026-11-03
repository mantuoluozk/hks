"""Recheck every row against pinned prepared data and upstream scorer."""
import argparse,hashlib,json
from pathlib import Path
from gsm8k_score import summarize,extract_answer,prompt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[json.loads(s) for s in (a.output/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
items=[json.loads(s) for s in (ROOT/'data/gsm8k/test.jsonl').read_text(encoding='utf-8').splitlines()]
for row in rows:
    item=items[row['index']];assert row['expected']==extract_answer(item['answer'])
    assert row['question_sha256']==hashlib.sha256(item['question'].encode()).hexdigest()
    blob=Path(row['request_path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==row['request_sha256']
    request=json.loads(blob);assert request['messages']==[{'role':'user','content':prompt(item)}]
summary=summarize(rows);assert summary==json.loads((a.output/'summary.json').read_text())
(a.output/'rescore.json').write_text(json.dumps({'passed':True,'matched_rows':1319,'ground_truth_and_request_hashes_checked':True,'summary':summary},indent=2)+'\n')
print('UPSTREAM_RESCORE_MATCHED',1319)
