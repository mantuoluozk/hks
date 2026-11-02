"""Export the verified optimization comparison without running inference."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METRICS=('duration','request_throughput','output_throughput','total_throughput','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_latency_ms')
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
 job=ROOT/'results'/Path(args.job).name
 data=json.loads((job/'verified-summary.json').read_text(encoding='utf-8'));assert data['complete']
 bench=ROOT/'bench';bench.mkdir(exist_ok=True)
 for repeat in (1,2):shutil.copyfile(job/f'process1/round{repeat}.jsonl',bench/f'round{repeat}.jsonl')
 (bench/'comparison.json').write_text(json.dumps(data,indent=2),newline="\n")
 lines=['# Recon vLLM 优化版v1完整矩阵','',
  '每档两次取第二次；单服务共608请求。基准为原版process1第二次；单位：duration秒、延迟毫秒、吞吐token/s或请求/s。独立进程复现：未在本轮安排，字段为null。','']
 for name,label in [('baseline','原版'),('optimized','优化版'),('change_percent','变化百分比（优化版/原版−1）')]:
  lines+=['## '+label,'','| 输入长度参数 | 并发 | '+' | '.join(METRICS)+' |','|'+'---|'*(len(METRICS)+2)]
  for r in data['comparison']:lines.append('| '+str(r['input'])+' | '+str(r['concurrency'])+' | '+' | '.join(f'{r[name][m]:.4f}' for m in METRICS)+' |')
  lines.append('')
 (bench/'summary.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
 print('Exported',bench)
if __name__=='__main__':main()
