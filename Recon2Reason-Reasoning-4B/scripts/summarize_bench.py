"""Build a concise table from preserved second-round benchmark JSONL."""
import json,sys
from pathlib import Path

directory=Path(sys.argv[1])
rows=[json.loads(line) for line in (directory/'round2.jsonl').read_text().splitlines() if line.strip()]
cols=['random_input_len','random_output_len','max_concurrency','completed','duration',
      'output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','mean_e2e_latency_ms']
lines=['# Diagnostic benchmark (not an official acceptance verdict)','',
       'See config.txt, env.sh and both raw JSONL/log files for the exact workload.','',
       '| '+' | '.join(cols)+' |','|'+' --- |'*len(cols)]
for row in rows:
    values=[]
    for key in cols:
        value=row.get(key,'-')
        values.append(f'{value:.2f}' if isinstance(value,float) else str(value))
    lines.append('| '+' | '.join(values)+' |')
(directory/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
