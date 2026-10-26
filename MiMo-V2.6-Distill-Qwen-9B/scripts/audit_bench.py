"""Validate actual request/token counts and unique cases in both rounds."""
import argparse,json,math,os
from pathlib import Path

def validate(rows,profile):
    inputs,concurrencies,output=([512],[1,4],128) if profile=='smoke' else ([4096,16384],[1,8,64],1024)
    expected={(i,c) for i in inputs for c in concurrencies};seen=set()
    for row in rows:
        key=(row['random_input_len'],row['max_concurrency'])
        assert key in expected and key not in seen,f'Unexpected/duplicate case: {key}'
        seen.add(key);n=max(8,key[1]*2)
        assert row['random_output_len']==output
        assert row['dataset_name']=='random-ids'
        assert row['completed']==n,(key,'completed',row['completed'],n)
        assert row['total_input_tokens']==key[0]*n,(key,'input tokens')
        assert row['total_output_tokens']==output*n,(key,'output tokens')
        for metric in ['duration','output_throughput','mean_ttft_ms','mean_tpot_ms']:
            assert math.isfinite(row[metric]) and row[metric]>0,(key,metric)
    assert seen==expected,f'Missing cases: {expected-seen}'
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--profile',choices=['smoke','matrix'],required=True)
    args=ap.parse_args();rounds={};errors=[]
    for number in (1,2):
        try:
            rows=[json.loads(line) for line in (args.folder/f'round{number}.jsonl').read_text().splitlines() if line.strip()]
            rounds[number]=validate(rows,args.profile)
        except (AssertionError,KeyError,ValueError,OSError) as exc:errors.append(f'round{number}: {exc}')
    summary={'scope':os.environ.get('EXECUTION_PLATFORM','self-owned BW100')+' self-test; not official acceptance','profile':args.profile,'complete':not errors,'errors':errors,
             'cases':sum(len(rows) for rows in rounds.values())}
    if not errors:
        fields=['random_input_len','random_output_len','max_concurrency','completed','duration','output_throughput',
                'mean_ttft_ms','mean_tpot_ms','total_output_tokens','total_output_tokens_retokenized']
        summary['round2']=[{k:r.get(k) for k in fields} for r in rounds[2]]
        lines=['| 输入 | 输出 | 并发 | 请求数 | 输出 token/s | TTFT ms | TPOT ms |', '|---:|---:|---:|---:|---:|---:|---:|']
        for r in sorted(rounds[2],key=lambda r:(r['random_input_len'],r['max_concurrency'])):
            lines.append(f'| {r["random_input_len"]} | {r["random_output_len"]} | {r["max_concurrency"]} | {r["completed"]} | {r["output_throughput"]:.2f} | {r["mean_ttft_ms"]:.2f} | {r["mean_tpot_ms"]:.2f} |')
        (args.folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (args.folder/'audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));raise SystemExit(bool(errors))
if __name__=='__main__':main()
