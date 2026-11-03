"""Read-only status; never start, stop, or resume a model task."""
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
current=json.loads((root/'results/formal-current.json').read_text(encoding='utf-8'))
job=Path(current['path']);state=json.loads((job/'state.json').read_text(encoding='utf-8'))
rows=[]
for p in (job/'process1').glob('round[12].jsonl'):
    rows.extend(json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip())
result={'job':job.name, 'state':state,'completed_cells':len(rows),
        'completed_counted_requests':sum(r['completed'] for r in rows),
        'cells':[{'input':r['random_input_len'],'concurrency':r['max_concurrency'],
                  'requests':r['completed'],'output_tokens_per_second':r['output_throughput']} for r in rows]}
if state.get('status')=='benchmarking':
    p=job/'process1'/f"round{state['repeat']}_{state['input_tokens']}_{state['output_tokens']}_c{state['concurrency']}.log"
    if p.exists():
        hits=re.findall(r'(\d+)/(\d+)\s+\[',p.read_text(encoding='utf-8',errors='replace'))
        result['active_cell_progress']=hits[-1] if hits else None
print(json.dumps(result))
