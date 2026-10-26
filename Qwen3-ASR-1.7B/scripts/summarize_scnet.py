"""Audit primary and fresh-image ASR evidence without declaring official accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def audit(folder):
    state = read(folder/'status.json')
    if state['stage'] != 'platform_tests_complete':
        assert read(folder/'recovery-status.json')['stage'] == 'platform_tests_complete'
    smoke = read(folder/'smoke/summary.json')
    assert smoke['functional_cases'] == 6 and all(smoke['cross_interface_exact'].values())
    bench = read(folder/'bench/summary.json')
    rows = [json.loads(x) for x in (folder/'bench/requests.jsonl').read_text().splitlines()]
    assert len(bench['workloads']) == 8 and bench['counted_requests'] == len(rows) == 128
    for workload in bench['workloads']:
        selected = [x for x in rows if x['round'] == workload['round'] and x['concurrency'] == workload['concurrency']]
        assert len(selected) == workload['requests']
        audio = sum(x['audio_seconds'] for x in selected)
        assert abs(audio/workload['wall_seconds'] - workload['audio_seconds_per_wall_second']) < 1e-8
    quality = read(folder/'accuracy/summary.json')
    predictions = [json.loads(x) for x in (folder/'accuracy/predictions.jsonl').read_text().splitlines()]
    assert quality['complete'] and quality['completed'] == quality['requested'] == len(predictions) == 2620
    assert len({x['id'] for x in predictions}) == 2620
    assert not quality['failures'] and quality['truncated'] == 0
    assert not any(x['truncated'] for x in predictions)
    wer = 100*sum(x['word_errors'] for x in predictions)/sum(x['reference_words'] for x in predictions)
    assert abs(wer-quality['wer_percent']) < 1e-10
    stream = read(folder/'input-streaming/summary.json')
    assert stream['completed'] == len(stream['cases']) == 4
    assert {(x['language_input'],x['step_ms']) for x in stream['cases']} == {('en',1000),('en',4000),('zh',1000),('zh',4000)}
    assert all(x['final_text'].strip() for x in stream['cases'])
    weights = read(folder/'runtime/weights-check.json')
    assert weights['all_required_files_match'] and len(weights['files']) == 11
    runtime = read(folder/'runtime/runtime-config.json')
    assert runtime['HIP_VISIBLE_DEVICES'] == '0' and runtime['ENFORCE_EAGER'] == '0'
    return {'functional_cases':6,'pcm_input_cases':4,'public_wer_percent':wer,
            'public_cer_percent':quality['cer_percent'],'public_cases':2620,
            'requests':128,'workloads':bench['workloads'],'runtime':runtime,
            'environment':read(folder/'runtime/environment.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', required=True)
    parser.add_argument('--reproduction', required=True)
    args = parser.parse_args()
    primary = ROOT/args.primary;repro = ROOT/args.reproduction
    first = audit(primary);second = audit(repro)
    for key in ['MEM_FRAC','MAX_LEN','MAX_NUM_SEQS','ENFORCE_EAGER']:
        assert first['runtime'][key] == second['runtime'][key], key
    assert first['environment']['devices'] == second['environment']['devices'], 'Different device specification'
    for key in ['vllm','transformers','openai']:
        assert first['environment']['packages'][key] == second['environment']['packages'][key], key
    baked = read(repro/'runtime/image-dependencies.json')
    assert baked['all_files_match_pinned_wheels'] and len(baked['packages']) == 8
    assert second['runtime']['ASR_EXTRA_DIR'] == '/opt/qwen3-asr-deps'
    gains = []
    for before in first['workloads']:
        if before['round'] != 2:continue
        after = next(x for x in second['workloads'] if x['round'] == 2 and x['concurrency'] == before['concurrency'])
        delta = 100*(after['audio_seconds_per_wall_second']/before['audio_seconds_per_wall_second']-1)
        gains.append({'concurrency':before['concurrency'],'primary_audio_seconds_per_wall_second':before['audio_seconds_per_wall_second'],
                      'reproduction_audio_seconds_per_wall_second':after['audio_seconds_per_wall_second'],'deviation_percent':delta})
    references = []
    for folder in [primary,repro]:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.jsonl','.log'} or 'backup' in path.name:continue
            references.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {'model':'Qwen/Qwen3-ASR-1.7B','evidence_audit_passed':True,
               'primary_path':args.primary,'reproduction_path':args.reproduction,
               'primary':first,'reproduction':second,'performance_reproduction':gains,
               'maximum_absolute_performance_deviation_percent':max(abs(x['deviation_percent']) for x in gains),
               'performance_reproduction_within_twenty_percent':all(abs(x['deviation_percent']) <= 20 for x in gains),
               'official_accuracy_pass':None,'official_acceptance_passed':None,
               'image':read(ROOT/'sources/scnet-image.json'),'references':references}
    (ROOT/'results/scnet/validation-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in {'primary','reproduction','references'}},indent=2,ensure_ascii=False))


if __name__ == '__main__':main()
