"""Audit published evidence without loading a model or changing measurements."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (a, b)

def main():
    manifest = read('results/evidence-manifest.json')
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    base = 'results/seedtts-full-seed1234/generation/'
    meta = read(base+'manifest.json')
    scores = lines(ROOT/(base+'public/scored.jsonl'))
    generation = lines(ROOT/(base+'public/generated.jsonl'))
    assert len(scores) == len(generation) == meta['selected_rows'] == meta['total_rows'] == 1088
    assert len(set(meta['ids'])) == 1088 and meta['full_dataset']
    for rows in [scores, generation]:
        assert len({r['id'] for r in rows}) == 1088
        assert {r['id'] for r in rows} == set(meta['ids'])
        assert all(r['passed'] for r in rows)
    summary = read(base+'wer-summary.json')
    wer = statistics.mean(r['wer'] for r in scores)*100
    close(wer, summary['mean_sample_wer_percent'])
    assert summary['errors'] == 0 and summary['scored'] == 1088 and not summary['filter_high_error_samples']
    assert summary['wer_above_50_count'] == sum(r['wer'] > .5 for r in scores)
    perf = read('results/performance-fp32-20261028/state.json')
    assert perf['complete'] and len(perf['rounds']) == 2
    metric_values = []
    wave_hashes = []
    for label, round_data in zip(['first', 'repeat'], perf['rounds']):
        prefix = 'results/performance-fp32-20261028/'+label
        smoke = read(prefix+'/smoke/summary.json')
        assert len(smoke['cases']) == 9 and smoke['passed']
        assert all(r['passed'] for r in smoke['cases'])
        assert smoke['checks']['same_seed_pcm_equal'] and smoke['checks']['different_seed_pcm_differs']
        wave_hashes.append({r['name']:r['waveform_sha256'] for r in smoke['cases'] if 'waveform_sha256' in r})
        bench = read(prefix+'/bench/summary.json')
        assert bench == round_data['performance'] and bench['complete'] and len(bench['cells']) == 4
        for cell in bench['cells']:
            d, c = cell['requested_seconds'], cell['concurrency']
            rows = [read(prefix+f'/bench/s{d}-c{c}-measure{i}.json') for i in range(8)]
            warm = [read(prefix+f'/bench/s{d}-c{c}-warm{i}.json') for i in range(c)]
            assert cell['requests'] == 8 and cell['warmup_requests'] == c
            assert all(r['status'] == 200 and not r['warmup'] for r in rows)
            assert all(r['status'] == 200 and r['warmup'] for r in warm)
            close(statistics.mean(r['latency_seconds'] for r in rows), cell['mean_latency_seconds'])
            close(sum(r['audio_seconds'] for r in rows)/cell['wall_seconds'], cell['audio_seconds_per_wall_second'])
            close(cell['mean_latency_seconds']/d, cell['mean_rtf'])
        metric_values.append(bench['cells'])
    assert wave_hashes[0] == wave_hashes[1] and len(wave_hashes[0]) == 7
    differences = [abs(b[k]-a[k])/a[k]*100 for a,b in zip(*metric_values)
                   for k in ['mean_latency_seconds','audio_seconds_per_wall_second']]
    close(max(differences), perf['maximum_difference_percent'])
    assert max(differences) <= 20
    image = read('results/new-image-quick-20261028/summary.json')
    assert image['passed'] and image['baked_files_verified'] == 588 and image['http_status'] == 200
    assert image['owned_server_stopped'] and image['samples'] == 96000 and image['sample_rate'] == 24000
    comparison = read('results/accuracy-comparison.json')
    close(comparison['measured_wer_percent'], wer)
    close(comparison['increase_percentage_points'], wer-1.02)
    assert comparison['tolerance_percentage_points'] == 2
    assert comparison['numeric_criterion_met'] == (wer <= 1.02+2)
    result = {'selfcheck_passed':True,'evidence_files_verified':len(manifest['sha256']),
              'full_dataset_rows':1088,'wer_percent':wer,'increase_percentage_points':wer-1.02,
              'criterion':'WER increase <= 2 percentage points','numeric_criterion_met':wer<=3.02,
              'maximum_mean_latency_and_throughput_deviation_percent':max(differences),
              'official_acceptance_claimed':False}
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
