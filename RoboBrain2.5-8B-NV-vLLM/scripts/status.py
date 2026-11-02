"""Compact read-only status of the current matrix."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];ptr=R/'results/formal-current.json'
if not ptr.exists():print(json.dumps({'status':'not_started'}));raise SystemExit
O=R/'results'/Path(json.loads(ptr.read_text(encoding='utf-8'))['path']).name
s=json.loads((O/'state.json').read_text(encoding='utf-8'));s['job']=O.name
rows=[]
for p in O.glob('round*.jsonl'):
 for line in p.read_text(encoding='utf-8').splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
s['completed_cells']=len(rows);s['completed_matrix_requests']=sum(r['completed'] for r in rows)
if s['status']=='benchmarking':
 p=O/f"round{s['repeat']}_{s['input_length']}_1024_c{s['concurrency']}.log"
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1024));tail=f.read().decode(errors='replace')
  parts=[x for x in re.split('[\r\n]',tail) if x.strip()];s['client_tail']=parts[-1][-180:] if parts else ''
print(json.dumps(s,ensure_ascii=False))
