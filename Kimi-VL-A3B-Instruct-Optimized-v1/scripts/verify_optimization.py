"""Recompute short-load throughput and compare saved responses, without a GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--repeats', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    job = ROOT / 'results' / Path(args.job).name
    runs = {}
    hashes = {}
    for name in ['baseline', 'decode-graph']:
        functional = read(job / name / 'functional/summary.json')
        assert functional['all_passed'] and functional['functional_passed'] == functional['functional_total'] == 8
        for repeat in range(1,args.repeats+1):
            folder = job / name / f'bench{repeat}'
            manifest = read(folder / 'summary.json')
            assert manifest['counted_requests'] == 32 and manifest['warmup_requests'] == 16
            for field in ['image_sha256', 'question', 'generation', 'script_sha256', 'http_client_sha256']:
                value = manifest[field]
                if field in hashes:
                    assert hashes[field] == value, field
                hashes[field] = value
            for cell in manifest['workloads']:
                c, o = cell['concurrency'], cell['max_tokens']
                assert c in [1, 2] and o in [32, 128]
                rows = [json.loads(x) for x in (folder / f'c{c}-o{o}/requests.jsonl').read_text().splitlines()]
                assert len(rows) == cell['requests'] == 8
                assert all(r['done_marker'] and r['output'] and r['finish_reason'] in ['stop', 'length'] for r in rows)
                tokens = sum(r['usage']['completion_tokens'] for r in rows)
                assert tokens == cell['completion_tokens']
                close(cell['output_tokens_per_second'], tokens / cell['wall_seconds'])
                close(cell['mean_latency_seconds'], statistics.mean(r['seconds'] for r in rows))
                close(cell['mean_ttft_seconds'], statistics.mean(r['ttft_seconds'] for r in rows))
                close(cell['mean_tpot_seconds'], statistics.mean((r['seconds'] - r['ttft_seconds']) / (r['usage']['completion_tokens'] - 1) for r in rows if r['usage']['completion_tokens'] > 1))
                runs[name, repeat, c, o] = (cell, rows)
    comparison = []
    for repeat in range(1,args.repeats+1):
        for c in [1, 2]:
            for o in [32, 128]:
                a, ar = runs['baseline', repeat, c, o]
                b, br = runs['decode-graph', repeat, c, o]
                comparison.append({'repeat': repeat, 'concurrency': c, 'max_tokens': o,
                                   'baseline_tps': a['output_tokens_per_second'], 'candidate_tps': b['output_tokens_per_second'],
                                   'gain_pct': 100 * (b['output_tokens_per_second'] / a['output_tokens_per_second'] - 1),
                                   'same_text': [r['output'] for r in ar] == [r['output'] for r in br],
                                   'same_output_lengths': [r['usage']['completion_tokens'] for r in ar] == [r['usage']['completion_tokens'] for r in br],
                                   'same_input_lengths': a['prompt_tokens'] == b['prompt_tokens']})
    stability = []
    for name in (['baseline', 'decode-graph'] if args.repeats==2 else []):
        for c in [1, 2]:
            for o in [32, 128]:
                a, _ = runs[name, 1, c, o]
                b, _ = runs[name, 2, c, o]
                stability.append({'candidate': name, 'concurrency': c, 'max_tokens': o,
                                  'throughput_deviation_pct': 100 * abs(b['output_tokens_per_second'] / a['output_tokens_per_second'] - 1)})
    report = {'job': job.name, 'comparison': comparison, 'stability': stability,
              'same_process_repetitions': args.repeats==2, 'measured_repeats':args.repeats, 'official_six_loads': False,
              'official_acceptance': None, 'protocol': hashes}
    (job / 'verified-optimization.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
