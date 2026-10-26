#!/usr/bin/env python3
"""Summarize SCNet validation evidence; never claims official accuracy."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM, STAGE = 'SCNet Notebook', 'platform_tests_complete'
def require(condition, message):
    if not condition:
        raise SystemExit(f'validation summary refused: {message}')
def inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), f'path outside model root: {path}')
    return path
def read(path):
    path = inside(path)
    require(path.is_file(), f'missing file: {path}')
    return path
def load(path):
    try:
        data = json.loads(read(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise SystemExit(f'validation summary refused: cannot read {path}: {exc}')
    require(isinstance(data, dict), f'expected JSON object: {path}')
    return data
def rows(path):
    try:
        return [json.loads(line) for line in read(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise SystemExit(f'validation summary refused: cannot read {path}: {exc}')
def artifact(state, key):
    artifacts = state.get('artifacts')
    require(isinstance(artifacts, dict), 'status.json has no artifacts object')
    rel = artifacts.get(key)
    require(isinstance(rel, str) and rel, f'missing artifact: {key}')
    path = inside(ROOT / rel)
    require(path.is_dir(), f'missing artifact directory: {key}')
    return path
def reference(path):
    path = read(path)
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
    require(len(sys.argv) == 2, 'usage: summarize_scnet.py <validation-evidence-dir>')
    raw = Path(sys.argv[1]); evidence = inside(raw if raw.is_absolute() else ROOT / raw)
    require(evidence.is_relative_to(ROOT / 'results/scnet'), 'evidence directory must be under results/scnet')
    state = load(evidence / 'status.json')
    require(state.get('stage') == STAGE, f"stage is {state.get('stage')!r}, not {STAGE!r}")
    require(state.get('execution_platform') == PLATFORM, 'status platform is not SCNet Notebook')
    runtime = artifact(state, 'runtime'); config = load(runtime / 'runtime-config.json')
    require(config.get('EXECUTION_PLATFORM') == PLATFORM, 'runtime platform is not SCNet Notebook')
    require(str(config.get('HIP_VISIBLE_DEVICES')) == '0', 'runtime logical card is not 0')
    require(config.get('CARD') == 'BW', 'runtime card is not BW')
    require(str(config.get('PRECISION') or '').lower() in {'bf16', 'bfloat16'}, 'runtime precision is not BF16')
    weights = load(runtime / 'weights-check.json'); files = weights.get('files')
    require(weights.get('all_required_files_match') is True, 'weights check did not pass')
    require(isinstance(files, list) and len(files) == 15, 'weights check does not list 15 files')
    require(all(isinstance(f, dict) and f.get('match') is True for f in files), 'not all runtime files match')
    smoke_dir = artifact(state, 'smoke'); smoke = load(smoke_dir / 'summary.json')
    require(smoke.get('passed') == 11 and smoke.get('total') == 11, 'smoke is not 11/11')
    bench = artifact(state, 'benchmark_matrix'); audit = load(bench / 'audit.json')
    require(audit.get('complete') is True and audit.get('cases') == 12, 'benchmark audit is not complete/12 cases')
    rounds = [rows(bench / f'round{n}.jsonl') for n in (1, 2)]
    require(all(len(group) == 6 for group in rounds), 'benchmark rounds must each have 6 cases')
    require(all(isinstance(r, dict) and type(r.get('completed')) is int for group in rounds for r in group), 'invalid rows')
    requests = sum(r['completed'] for group in rounds for r in group)
    require(requests == 608, f'benchmark rounds total {requests} requests, not 608')
    quality = artifact(state, 'quality'); datasets = load(quality / 'summary.json').get('datasets') or {}
    require(isinstance(datasets, dict), 'quality summary has no datasets object')
    scores = {}
    for name in ('gsm8k', 'mmlu'):
        row = datasets.get(name)
        require(isinstance(row, dict) and row.get('total') == 200 and row.get('errors') == 0, f'{name} is not 200 samples/0 errors')
        correct = row.get('correct')
        require(type(correct) is int and 0 <= correct <= 200, f'{name} correct count is invalid')
        scores[name] = {'correct': correct, 'total': 200, 'accuracy_percent': round(100*correct/200, 2),
                        'errors': 0, 'truncated': row['truncated'], 'unparsed': row['unparsed'],
                        'scope': 'fixed 200 public samples per dataset; not official or full test split'}
    predictions = rows(quality / 'predictions.jsonl')
    require(len(predictions) == 400, 'quality predictions must contain 400 rows')
    for name in ('gsm8k', 'mmlu'):
        subset = [r for r in predictions if r.get('dataset') == name]
        require(len(subset) == 200 and len({r['index'] for r in subset}) == 200, f'{name} has missing/duplicate predictions')
        require(all(r.get('error') is None for r in subset), f'{name} has request errors')
        require(sum(r.get('correct') is True for r in subset) == scores[name]['correct'], f'{name} count differs from raw predictions')
        require(sum(r.get('truncated') is True for r in subset) == scores[name]['truncated'], f'{name} truncation count mismatch')
        require(sum(r.get('predicted') is None for r in subset) == scores[name]['unparsed'], f'{name} parsing count mismatch')
    fields = ('random_input_len', 'random_output_len', 'max_concurrency', 'completed', 'duration',
              'output_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'total_output_tokens')
    require(all(k in r for r in rounds[1] for k in fields), 'round2 rows are missing performance metrics')
    performance = [{k: r[k] for k in fields} for r in rounds[1]]
    performance.sort(key=lambda item: (item['random_input_len'], item['max_concurrency']))
    paths = [evidence/'status.json', runtime/'runtime-config.json', runtime/'weights-check.json', smoke_dir/'summary.json',
             bench/'audit.json', bench/'round1.jsonl', bench/'round2.jsonl', quality/'summary.json', quality/'manifest.json',
             quality/'predictions.jsonl', runtime/'environment.json', evidence/'framework-source-check.json']
    references = [reference(path) for path in paths]
    output = {'scope': 'SCNet Notebook platform self-test evidence; not official accuracy or acceptance',
              'evidence_status': 'PASS', 'evidence_dir': evidence.relative_to(ROOT).as_posix(),
              'runtime': {'execution_platform': PLATFORM, 'logical_device': 0, 'card': 'BW',
                          'precision': 'BF16', 'model': config.get('MODEL_ID'),
                          'image': config.get('IMAGE'), 'weights_verified_files': len(files)},
              'functional': {'smoke': {'passed': smoke['passed'], 'total': smoke['total']}},
              'performance': {'benchmark': 'matrix', 'round': 2, 'requests_both_rounds': requests,
                              'metrics': performance},
              'public_self_eval': scores, 'references': references,
              'official_accuracy_pass': None, 'official_acceptance_passed': None}
    target = inside(evidence / 'validation-summary.json'); temp = inside(evidence / 'validation-summary.tmp')
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temp.replace(target); print(target)
if __name__ == '__main__':
    main()
