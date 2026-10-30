"""Frozen full-set reevaluation; preserve old scores and retain complete API responses."""
import argparse
import ast
import base64
import concurrent.futures
import hashlib
import json
import re
import shutil
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'ocrbench': 'OCRBench', 'mmmu': 'MMMU', 'chartqa': 'ChartQA'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_parts(name, item, data):
    if name == 'mmmu':
        text = item['question']
        if item['question_type'] == 'multiple-choice':
            text += '\nOptions:\n' + '\n'.join(f'({chr(65+i)}) {v}' for i, v in enumerate(item['options']))
            text += '\nAnswer with the option letter only.'
        else:
            text += '\nAnswer the question using a single word or phrase.'
        paths = {int(re.search(r'_(\d+)\.png$', p).group(1)): data / 'img' / p for p in item['img_paths']}
        refs = [int(v) for v in re.findall(r'<image\s+(\d+)>', text)]
        assert set(refs) <= set(paths), (item['id'], refs, list(paths))
        if not refs:
            return [('image', p) for p in paths.values()] + [('text', text)]
        parts = []
        for v in re.split(r'(<image\s+\d+>)', text):
            match = re.fullmatch(r'<image\s+(\d+)>', v)
            if match:
                parts.append(('image', paths[int(match.group(1))]))
            elif v:
                parts.append(('text', v))
        return parts
    text = item['question']
    if name == 'chartqa':
        text += '\nAnswer the question using a single word or phrase.'
    return [('image', data / item['img_path']), ('text', text)]


def scorer(path, name):
    # Freeze and reuse precisely the old metric to separate generation changes
    # from scoring changes. No code from main() or command-line assignments runs.
    tree = ast.parse(path.read_text())
    ns = {'re': re, 'unicodedata': unicodedata}
    exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)], type_ignores=[]), str(path), 'exec'), ns)
    return ns[{'ocrbench': 'get_score', 'chartqa': 'relaxed_correctness', 'mmmu': 'score'}[name]]


def run(name, job, raw_root, workers, base, indices=None, mode='4x'):
    data = ROOT / 'data' / DATASETS[name]
    dataset_path = data / 'data.json'
    items = json.loads(dataset_path.read_text())
    assert len(items) == {'ocrbench': 1000, 'mmmu': 900, 'chartqa': 2500}[name]
    assert len({v['question_id'] for v in items}) == len(items)
    if indices is not None:
        items = [items[i] for i in indices]
    out = job / name
    out.mkdir(parents=True, exist_ok=False)
    raw = raw_root / name
    raw.mkdir(parents=True, exist_ok=False)
    score = scorer(job / 'source' / f'eval_{name}.py', name)
    protocol = {'dataset': name, 'rows': len(items), 'indices': indices, 'dataset_json_sha256': digest(dataset_path),
                'thinking': False, 'downsample_mode': mode, 'max_slice_nums': 9, 'temperature': 0,
                'max_tokens': 1024, 'workers': workers, 'base': base, 'seed': 42,
                'scorer_sha256': digest(job / 'source' / f'eval_{name}.py'),
                'official_protocol_fully_known': False, 'scoring_changed_from_dsh': False,
                'mmmu_options': 'explicit labels, image placeholders interleaved', 'retry_policy': 'none'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # Validate every image reference before sending the first question.
    prepared = []
    for item in items:
        parts = prompt_parts(name, item, data)
        for kind, value in parts:
            if kind == 'image':
                assert value.is_file(), value
        prepared.append((item, parts))

    def ask(pair):
        item, parts = pair
        content, audit = [], []
        for kind, value in parts:
            if kind == 'image':
                blob = value.read_bytes()
                audit.append({'type': 'image', 'path': str(value), 'sha256': hashlib.sha256(blob).hexdigest()})
                content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(blob).decode()}})
            else:
                audit.append({'type': 'text', 'text': value})
                content.append({'type': 'text', 'text': value})
        body = {'model': 'minicpm-v-4.7', 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'seed': 42, 'max_tokens': 1024, 'stream': False,
                'chat_template_kwargs': {'enable_thinking': False},
                'mm_processor_kwargs': {'downsample_mode': mode, 'max_slice_nums': 9}}
        started = time.perf_counter()
        record = {'question_id': item['question_id'], 'id': item.get('id'),
                  'type': item.get('question_type', item.get('qtype')), 'subject': item.get('subject'),
                  'answer': item['answer'], 'request_audit': audit, 'request_parameters': {k: v for k, v in body.items() if k != 'messages'}}
        try:
            req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=900) as r:
                response = json.loads(r.read().decode())
            choice = response['choices'][0]
            text = choice['message'].get('content') or ''
            record.update(response=response, prediction=text, finish_reason=choice['finish_reason'], usage=response.get('usage'),
                          score=score(item['answer'], text) if name == 'chartqa' else score(item, text))
        except Exception as e:
            record.update(error=repr(e), score=0, prediction='')
            if isinstance(e, urllib.error.HTTPError):
                record['http_error_body'] = e.read().decode(errors='replace')
        record['seconds'] = time.perf_counter() - started
        return record

    rows = []
    with (raw / 'responses.jsonl').open('w') as f, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, value) for value in prepared]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            rows.append(row)
            summary = {'dataset': name, 'completed': len(rows), 'expected': len(items), 'correct': sum(r['score'] for r in rows),
                       'http_errors': sum('error' in r for r in rows), 'length': sum(r.get('finish_reason') == 'length' for r in rows),
                       'empty': sum(not r['prediction'].strip() for r in rows), 'official_protocol_fully_known': False}
            (out / 'progress.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert len(rows) == len(items) and {r['question_id'] for r in rows} == {i['question_id'] for i in items}
    summary.update(accuracy_x100=100 * summary['correct'] / len(items), responses_sha256=digest(raw / 'responses.jsonl'),
                   raw_path=str(raw), all_ids_verified=True, full_dataset=indices is None)
    by_type = {}
    for r in rows:
        key = r['subject'] or r['type']
        slot = by_type.setdefault(key, [0, 0])
        slot[0] += r['score']
        slot[1] += 1
    summary['by_type'] = by_type
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=DATASETS)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--base', default='http://127.0.0.1:30147')
    p.add_argument('--indices')
    a = p.parse_args()
    run(a.dataset, a.job, a.raw, a.workers, a.base, [int(i) for i in a.indices.split(',')] if a.indices else None)
