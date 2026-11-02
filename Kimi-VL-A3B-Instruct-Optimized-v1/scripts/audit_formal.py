"""Independently audit saved six-load counts, throughput and TTFT statistics."""
import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), (a, b)


def percentile(values, pct):
    s = sorted(values)
    x = (len(s) - 1) * pct / 100
    i = int(x)
    return s[i] + (s[min(i + 1, len(s) - 1)] - s[i]) * (x - i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = ROOT / 'results' / Path(args.job).name
    verified = json.loads((job / 'verified-summary.json').read_text(encoding='utf-8'))
    assert verified['complete']
    cells = []
    for process in range(1, verified['processes'] + 1):
        for repeat in [1, 2]:
            rows = [json.loads(x) for x in (job / f'process{process}/round{repeat}.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(rows) == 6
            for row in rows:
                c = row['max_concurrency']
                n = max(8, 2 * c)
                assert row['completed'] == len(row['output_lens']) == len(row['errors']) == n
                assert not any(row['errors']) and set(row['output_lens']) == {1024}
                assert all(x.strip() for x in row['generated_texts'])
                assert row['total_output_tokens'] == sum(row['output_lens']) == n * 1024
                assert row['total_input_tokens'] == sum(row['input_lens'])
                close(row['request_throughput'], n / row['duration'])
                close(row['output_throughput'], sum(row['output_lens']) / row['duration'])
                close(row['total_throughput'], (sum(row['input_lens']) + sum(row['output_lens'])) / row['duration'])
                close(row['mean_ttft_ms'], statistics.mean(row['ttfts']) * 1000)
                close(row['p99_ttft_ms'], percentile(row['ttfts'], 99) * 1000)
                # All output lengths are 1024: an independent aggregate consistency identity.
                close(row['mean_e2e_latency_ms'], row['mean_ttft_ms'] + 1023 * row['mean_tpot_ms'])
                for key in ['mean_tpot_ms', 'p99_tpot_ms', 'mean_e2e_latency_ms']:
                    assert math.isfinite(row[key]) and row[key] > 0
                cells.append({'process': process, 'repeat': repeat, 'input': row['random_input_len'],
                              'concurrency': c, 'requests': n, 'duration': row['duration']})
    report = {'job': job.name, 'verified_cells': len(cells), 'requests': sum(x['requests'] for x in cells),
              'cells': cells, 'throughput_and_ttft_recomputed': True,
              'mean_tpot_e2e_identity_checked': True,
              'latency_limit': 'The stock output-details JSONL omits individual end-to-end latencies. TPOT/E2E aggregates are consistency-checked; p99 TPOT is retained from the client, not independently reconstructed.',
              'official_acceptance': None}
    backend = []
    for process in range(1,verified['processes']+1):
        folder=job/f'process{process}'
        metrics=(folder/'round2_16384_1024_c64.server-metrics.prom').read_text(encoding='utf-8')
        def counter(name):
            values=[float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                    if line.startswith(name+'{') or line.startswith(name+' ')]
            assert values,name
            return sum(values)
        functional=sum(json.loads((folder/f'functional/{name}.json').read_text(encoding='utf-8'))['usage']['completion_tokens']
                       for name in ['text','red','blue-zh','ocr','demo','two-images','repeat','stream'])
        expected=608*1024+12*32+functional
        actual=counter('vllm:generation_tokens_total')
        assert actual==expected,(actual,expected)
        successful=counter('vllm:request_success_total')
        assert successful==608+12+8,successful
        backend.append({'process':process,'generation_tokens':actual,'expected_generation_tokens':expected,
                        'counted_requests':608,'client_warmups':12,'warmup_output_tokens_each':32,
                        'functional_requests':8,'functional_output_tokens':functional,'successful_requests':successful})
    report['backend_counters']=backend
    (job / 'independent-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
