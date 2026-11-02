"""Export second-repetition official performance values after successful audits."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--job', required=True)
args = parser.parse_args()
job = ROOT / 'results' / Path(args.job).name
verified = json.loads((job / 'verified-summary.json').read_text())
audit = json.loads((job / 'independent-audit.json').read_text())
assert verified['complete'] and audit['requests'] == verified['processes'] * 608
rows = [json.loads(x) for x in (job / 'process1/round2.jsonl').read_text().splitlines()]
assert len(rows) == 6
fields = ['random_input_len', 'random_output_len', 'max_concurrency', 'completed', 'duration',
          'request_throughput', 'output_throughput', 'total_throughput', 'mean_ttft_ms',
          'p99_ttft_ms', 'mean_tpot_ms', 'p99_tpot_ms', 'mean_e2e_latency_ms']
lines = ['| ' + ' | '.join(fields) + ' |', '|' + '|'.join(['---'] * len(fields)) + '|']
for row in rows:
    lines.append('| ' + ' | '.join(str(row[k]) if isinstance(row[k], int) else f'{row[k]:.4f}' for k in fields) + ' |')
lines += ['', 'duration单位为秒，延迟单位为毫秒；吞吐单位分别为请求/秒、输出token/秒、总token/秒。', '',
          '| 文本长度参数 | 并发 | 客户端processor输入token范围（含模板与图像） |', '|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['random_input_len']} | {row['max_concurrency']} | {min(row['input_lens'])}–{max(row['input_lens'])} |")
text = '\n'.join(lines) + '\n'
(job / 'performance-table.md').write_text(text, encoding='utf-8', newline='\n')
print(text)
