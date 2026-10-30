"""Independently recompute a completed OCRBench run from its saved API responses."""
import argparse
import ast
import hashlib
import json
import re
import unicodedata
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('job', type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
summary = json.loads((a.job / 'ocrbench/summary.json').read_text())
protocol = json.loads((a.job / 'ocrbench/protocol.json').read_text())
raw = Path(summary['raw_path']) / 'responses.jsonl'
assert hashlib.sha256(raw.read_bytes()).hexdigest() == summary['responses_sha256']
rows = [json.loads(s) for s in raw.read_text().splitlines()]
items = json.loads((root / 'data/OCRBench/data.json').read_text())
gold = {v['question_id']: v for v in items}
assert len(rows) == len(gold) == 1000
assert len({v['question_id'] for v in rows}) == 1000
assert set(gold) == {v['question_id'] for v in rows}
source = a.job / 'source/eval_ocrbench.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['scorer_sha256']
ns = {'re': re, 'unicodedata': unicodedata}
tree = ast.parse(source.read_text())
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)], type_ignores=[]), str(source), 'exec'), ns)
old = {r['question_id']: r for r in [json.loads(s) for s in (root / 'results/accuracy/ocrbench/predictions.jsonl').read_text().splitlines()]}
differences = []
counts = {'correct': 0, 'http_errors': 0, 'length': 0, 'empty': 0, 'over_64_tokens': 0, 'over_64_tokens_correct': 0,
          'wrong_to_right': 0, 'right_to_wrong': 0, 'changed_text': 0}
plain_correct = old_plain_correct = 0
max_prompt = max_completion = 0
for row in rows:
    item = gold[row['question_id']]
    assert row['answer'] == item['answer']
    params = row['request_parameters']
    assert params['temperature'] == 0 and params['max_tokens'] == 1024
    assert params['chat_template_kwargs'] == {'enable_thinking': False}
    assert params['mm_processor_kwargs'] == {'downsample_mode': '16x', 'max_slice_nums': 9}
    assert row['request_audit'][1]['text'] == item['question']
    score = ns['get_score'](item, row['prediction']) if 'error' not in row else 0
    assert score == row['score']
    if 'error' not in row:
        response = row['response']['choices'][0]
        assert (response['message'].get('content') or '') == row['prediction']
        assert response['finish_reason'] == row['finish_reason']
        usage = row['usage']
        max_prompt = max(max_prompt, usage['prompt_tokens'])
        max_completion = max(max_completion, usage['completion_tokens'])
        counts['over_64_tokens'] += usage['completion_tokens'] > 64
        counts['over_64_tokens_correct'] += int(usage['completion_tokens'] > 64 and score)
    # Also report the original publisher's plain whitespace/case normalization;
    # never choose between metrics based on which score is higher.
    def plain(value):
        if item['dataset'] == 'HME100k':
            return value.strip().replace('\n', ' ').replace(' ', '')
        return value.lower().strip().replace('\n', ' ')
    answers = item['answer'] if isinstance(item['answer'], list) else [item['answer']]
    plain_correct += int('error' not in row and any(plain(x) in plain(row['prediction']) for x in answers))
    counts['correct'] += score
    counts['http_errors'] += 'error' in row
    counts['length'] += row.get('finish_reason') == 'length'
    counts['empty'] += not row['prediction'].strip()
    previous = old[row['question_id']]
    old_plain_correct += int(any(plain(x) in plain(previous['prediction']) for x in answers))
    counts['wrong_to_right'] += not previous['score'] and bool(score)
    counts['right_to_wrong'] += bool(previous['score']) and not score
    counts['changed_text'] += previous['prediction'] != row['prediction']
    if score != previous['score']:
        differences.append({'question_id': row['question_id'], 'old': previous['score'], 'new': score,
                            'new_completion_tokens': (row.get('usage') or {}).get('completion_tokens')})
assert counts['correct'] == summary['correct']
result = {'passed': True, 'rows': 1000, **counts, 'accuracy_x100': counts['correct'] / 10,
          'publisher_plain_normalization_x100': plain_correct / 10, 'old_accuracy_x100': 84.8,
          'old_publisher_plain_normalization_x100': old_plain_correct / 10,
          'official_16x_instruct_reference': 90.40, 'gap_percentage_points': counts['correct'] / 10 - 90.4,
          'max_prompt_tokens': max_prompt, 'max_completion_tokens': max_completion,
          'raw_sha256': summary['responses_sha256'], 'score_changes': differences,
          'official_full_protocol_unknown': True}
(a.job / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'score_changes'}))
