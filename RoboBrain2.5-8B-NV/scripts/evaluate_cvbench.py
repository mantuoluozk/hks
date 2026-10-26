"""Full CV-Bench evaluation with pinned FlagEvalMM prompt and scoring functions.

Datasets and upstream evaluator snapshots live in cache and are not distributed.
No incorrect/empty/failed response is removed from the denominator.
"""
import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
import time
import typing
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/cvbench-protocol.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def upstream_functions(protocol):
    """Execute only the required functions from hash-verified upstream sources."""
    ns = dict(vars(typing), re=re)
    for item in protocol['upstream_files']:
        path = ROOT / 'cache/flagevalmm' / item['path']
        assert sha(path) == item['sha256'], path
        tree = ast.parse(path.read_text())
        if path.name == 'prompt_template.py':
            nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PromptTemplate']
            nodes[0].decorator_list = []
        elif path.name == 'vqa_base_dataset.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VqaBaseDataset')
            nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'build_prompt']
        elif path.name == 'pre_process.py':
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                     {'strip_answer', 'remove_special_characters', 'process_multiple_choice'}]
        else:
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--runtime-dir', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='Diagnostic only; zero runs the full test split')
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    protocol = json.loads(CONFIG.read_text())
    ns = upstream_functions(protocol)
    rows = []
    for item in protocol['dataset_files']:
        path = ROOT / 'cache/datasets/CV-Bench' / item['file']
        assert sha(path) == item['sha256'], path
        rows.extend(pq.read_table(path).to_pylist())
    assert len(rows) == 2638, len(rows)
    assert len({r['idx'] for r in rows}) == len(rows), 'Duplicate IDs'
    total_count = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    spec = {**protocol, 'workers': args.workers, 'limit': args.limit, 'expected_rows': len(rows),
            'full_test_rows': total_count, 'started_at': datetime.now(timezone.utc).isoformat(),
            'runner_sha256': sha(Path(__file__)), 'image_encoding': 'FlagEvalMM HttpClient image encoding: RGB JPEG quality=95; no manual resizing',
            'official_contest_accuracy_pass': None}
    spec['runtime'] = json.loads((args.runtime_dir/'runtime-config.json').read_text())
    spec['deterministic_patch'] = json.loads((args.runtime_dir/'deterministic-lds-patch.json').read_text())
    spec_path = args.output / 'protocol.json'
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        for k in ('runner_sha256', 'limit', 'generation', 'upstream_revision', 'dataset_revision', 'runtime', 'deterministic_patch'):
            assert old[k] == spec[k], ('Cannot mix protocols', k)
    else:
        spec_path.write_text(json.dumps(spec, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    annotations = {}
    for row in rows:
        annotations[str(row['idx'])] = {'question_id': row['idx'], 'question': row['question'],
                                      'question_type': 'multiple-choice', 'options': row['choices'],
                                      'answer': row['answer'][1], 'source': row['source']}
    prompt_builder = SimpleNamespace(prompt_template=ns['PromptTemplate']())
    predictions_path = args.output / 'predictions.jsonl'
    predictions = {}
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            p = json.loads(line)
            qid = str(p['question_id'])
            assert qid in annotations and qid not in predictions
            predictions[qid] = p

    def run(row):
        qid = str(row['idx'])
        prompt = ns['build_prompt'](prompt_builder, annotations[qid], ['image.png'])
        # FlagEvalMM HttpClient resolves the image placeholder to a multimodal part.
        # Its RoboBrain adapter appends this suffix before sending the request.
        prompt = prompt.replace('<image 1>', '') + '<think></think><answer>'
        image = row['image']
        raw_image = image['bytes'] if image['bytes'] is not None else Path(image['path']).read_bytes()
        im = Image.open(io.BytesIO(raw_image))
        im = im.convert('RGB')
        buf = io.BytesIO(); im.save(buf, format='JPEG', quality=95); png = buf.getvalue()
        payload = dict(protocol['generation'])
        payload.update(model='robobrain25-8b', messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(png).decode()}},
            {'type': 'text', 'text': prompt}]}])
        record = {'question_id': qid, 'source': row['source'], 'task': row['task'],
                  'prompt': prompt, 'label': annotations[qid]['answer'], 'image_sha256': hashlib.sha256(png).hexdigest(),
                  'image_size': list(im.size), 'answer': '', 'error': None}
        start = time.monotonic()
        try:
            req = urllib.request.Request('http://127.0.0.1:30125/v1/chat/completions',
                                         json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with opener.open(req, timeout=600) as response:
                result = json.loads(response.read())
            record['response'] = result
            choice = result['choices'][0]
            content = choice['message'].get('content') or ''
            if '</think>' in content:
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                content = match.group(1).strip() if match else ''
            record['answer'] = content
            record['finish_reason'] = choice.get('finish_reason')
            if choice.get('finish_reason') == 'length':
                record['error'] = 'truncated_response'
        except Exception as e:
            record['error'] = type(e).__name__ + ': ' + str(e)
        record['seconds'] = round(time.monotonic() - start, 3)
        return record

    pending = [r for r in rows if str(r['idx']) not in predictions]
    with predictions_path.open('a', encoding='utf-8') as dest, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, row) for row in pending]
        for future in as_completed(futures):
            record = future.result()
            predictions[record['question_id']] = record
            dest.write(json.dumps(record, ensure_ascii=False) + '\n'); dest.flush()
            if len(predictions) % 25 == 0 or record['error']:
                print(json.dumps({'completed':len(predictions), 'total':len(rows),
                                  'last_error':record['error']}, ensure_ascii=False), flush=True)
            if record['error'] and any(x in record['error'] for x in ('RemoteDisconnected', 'Connection refused')):
                for pending_future in futures:
                    pending_future.cancel()
                raise RuntimeError('Service unavailable; partial evidence preserved. Inspect runtime before continuing.')
    scored = []
    for row in rows:
        p = predictions[str(row['idx'])]
        pred = {'question_id': p['question_id'], 'answer': p['answer'] if not p['error'] else ''}
        scored.append(pred)
    metrics = ns['get_result'](annotations, scored)
    if len(rows) != total_count:
        metrics = {'diagnostic_micro_accuracy': round(sum(p['correct'] for p in scored) / len(scored) * 100, 2)}
    errors = sum(p['error'] is not None for p in predictions.values())
    summary = {'benchmark': 'CV-Bench', 'completed':len(predictions), 'expected':len(rows),
               'full_test': len(rows) == total_count, 'errors': errors, 'metrics':metrics,
               'publisher_nv8b_score':94.58, 'difference_percentage_points':round(metrics['accuracy']-94.58,2)
               if len(rows)==total_count else None,
               'comparison_status':'public_benchmark_retest; paper run config not published',
               'official_contest_accuracy_pass':None, 'predictions_sha256':sha(predictions_path),
               'finished_at':datetime.now(timezone.utc).isoformat()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'scored-predictions.json').write_text(json.dumps(scored,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
