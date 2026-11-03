"""CPU paired comparison of the same 200 GSM8K question prompts."""
import argparse,hashlib,json
from pathlib import Path
from prediction_io import read_predictions
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
baseline=json.loads((ROOT/'baselines/sglang-gsm8k-200.json').read_text());old={r['index']:r for r in baseline['rows']}
rows=read_predictions(a.output);assert len(rows)==200;comparison=[]
for r in rows:
    before=old[r['index']];request=json.loads(Path(r['request_path']).read_text())
    sha=hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest();assert sha==before['input_sha256']
    assert r['expected']==before['expected']
    comparison.append({'index':r['index'],'input_sha256':sha,'request_sha256':r['request_sha256'],
                       'sglang_correct':before['correct'],'vllm_correct':r['correct'],
                       'sglang_prediction':before['predicted'],'vllm_prediction':r['predicted']})
result={'same_prompt_hashes':200,'same_ground_truth':200,'sglang_correct':179,'vllm_correct':sum(r['correct'] for r in rows),
        'gained':sum(not r['sglang_correct'] and r['vllm_correct'] for r in comparison),
        'lost':sum(r['sglang_correct'] and not r['vllm_correct'] for r in comparison),
        'scope':'Paired fixed sample; different framework/image/date; no official acceptance claim','rows':comparison}
(a.output/'sglang-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
