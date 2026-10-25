"""Reproducible public OCRBench/MMMU evaluation (not official SCNet acceptance).

Uses pinned dataset bytes and vendored official scoring logic. Failed requests
remain in the denominator. Resume only when the entire run manifest matches.
"""
import argparse
import ast
import base64
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mmmu_score', ROOT/'sources/mmmu_eval_utils.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def load_samples(dataset):
    import pyarrow.parquet as pq
    meta = json.loads((ROOT/'sources/datasets-metadata.json').read_text())
    entry = next(d for d in meta if d['repo'] == ('echo840/OCRBench' if dataset == 'ocrbench' else 'MMMU/MMMU'))
    samples = []
    for item in entry['files']:
        path = ROOT/'data/public'/entry['repo']/item['path']
        assert path.stat().st_size == item['size'], path
        assert digest(path) == item['lfs']['oid'], path
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            row['_id'] = row.get('id', f'ocrbench_test_{index:04d}')
            row['_category'] = row.get('question_type') if dataset == 'ocrbench' else item['path'].split('/')[0]
            samples.append(row)
    assert len(samples) == (1000 if dataset == 'ocrbench' else 900), len(samples)
    assert len({s['_id'] for s in samples}) == len(samples)
    return samples, entry


def image_part(value):
    data = value['bytes']
    assert data, 'Dataset image bytes missing'
    with Image.open(io.BytesIO(data)) as im:
        mime = Image.MIME[im.format]
        size = im.size
    return {'type':'image_url', 'image_url':{'url':f'data:{mime};base64,'+base64.b64encode(data).decode()}}, {
        'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data), 'size':size}


def make_content(row, dataset):
    question = row['question']
    if dataset == 'ocrbench':
        part, info = image_part(row['image'])
        return [part, {'type':'text','text':question}], {'question':question, 'images':[info]}
    if row['question_type'] == 'multiple-choice':
        options = ast.literal_eval(row['options'])
        assert isinstance(options, list) and len(options) >= 2
        question += '\n' + '\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(options))
        question += '\nAnswer with the option letter only.'
    else:
        question += '\nAnswer the question using a single word or phrase.'
    parts, images = [], []
    tokens = re.split(r'(<image\s+\d+>)', question)
    for token in tokens:
        match = re.fullmatch(r'<image\s+(\d+)>', token)
        if match:
            key = 'image_'+match.group(1)
            assert row.get(key), (row['_id'], key)
            part, info = image_part(row[key])
            parts.append(part); images.append(dict(info, key=key))
        elif token:
            parts.append({'type':'text','text':token})
    if not images:
        for i in range(1,8):
            if row.get(f'image_{i}'):
                part, info = image_part(row[f'image_{i}'])
                parts.insert(0, part); images.append(dict(info, key=f'image_{i}'))
    assert images, row['_id']
    return parts, {'question':question, 'images':images}


def infer(row, args):
    start = time.monotonic()
    record = {'id':row['_id'], 'category':row['_category'], 'error':None}
    try:
        content, trace = make_content(row, args.dataset)
        record['request'] = trace
        payload = {'model':args.model, 'messages':[{'role':'user','content':content}],
                   'temperature':0, 'top_p':1, 'max_tokens':args.max_tokens, 'stream':False}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(args.url+'/v1/chat/completions', json.dumps(payload).encode(),
                                         {'Content-Type':'application/json'})
        with opener.open(request, timeout=1800) as response:
            raw = response.read().decode()
        record['raw_response'] = raw
        data = json.loads(raw)
        choice = data['choices'][0]
        record['output'] = choice['message'].get('content') or ''
        record['finish_reason'] = choice.get('finish_reason')
        record['usage'] = data.get('usage')
        if not record['output'].strip():
            record['error'] = 'empty content'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        record.setdefault('output', '')
    record['seconds'] = round(time.monotonic()-start, 3)
    return record


class TrackedRandom:
    def __init__(self, sample_id):
        self.rng = random.Random(int(hashlib.sha256(sample_id.encode()).hexdigest(),16))
        self.used = False

    def choice(self, options):
        self.used = True
        return self.rng.choice(options)


def grade(row, record, dataset):
    result = dict(record, gold=row.get('answer'), random_fallback=False, correct=False)
    if record.get('error'):
        result['strict_correct'] = False
        return result
    output = record['output']
    if dataset == 'ocrbench':
        # Official OCRBench example.py: HME100k removes spaces, preserves case.
        def normalize(text):
            text = text.strip().replace('\n',' ')
            return text.replace(' ','') if row['dataset']=='HME100k' else text.lower()
        answers = row['answer'] if isinstance(row['answer'],list) else [row['answer']]
        result['correct'] = any(normalize(a) in normalize(output) for a in answers)
        result['parsed'] = normalize(output)
    elif row['question_type'] == 'multiple-choice':
        options = ast.literal_eval(row['options'])
        mapping = {chr(65+i):s for i,s in enumerate(options)}
        rng = TrackedRandom(row['_id'])
        scorer.random = rng
        result['parsed'] = scorer.parse_multi_choice_response(output, list(mapping), mapping)
        result['random_fallback'] = rng.used
        result['correct'] = scorer.eval_multi_choice(row['answer'], result['parsed'])
    else:
        result['parsed'] = sorted(scorer.parse_open_response(output), key=str)
        result['correct'] = scorer.eval_open(row['answer'], result['parsed'])
    result['correct'] = bool(result['correct'])
    result['strict_correct'] = result['correct'] and not result['random_fallback']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ocrbench','mmmu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 means full dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--model', default='recon2reason-4b')
    parser.add_argument('--url', default='http://127.0.0.1:30123')
    args = parser.parse_args()
    samples, dataset_meta = load_samples(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset':dataset_meta, 'ids':[r['_id'] for r in samples],
                'generation':{'temperature':0,'top_p':1,'max_tokens':args.max_tokens},
                'prompt_protocol':'OCRBench original question; MMMU interleaved images/options, zero-shot letter/phrase v1',
                'model':args.model, 'url':args.url, 'workers':args.workers,
                'runtime':json.loads((ROOT/'results/runtime-config.json').read_text()),
                'evaluator_sha256':digest(Path(__file__)),
                'mmmu_scorer_sha256':digest(ROOT/'sources/mmmu_eval_utils.py'),
                'processor_sha256':digest(Path(os.environ.get('MODEL_PATH','/data1/model/Recon2Reason-Reasoning-4B'))/'preprocessor_config.json'),
                'model_revision':'f8faf41a1d5fe392f2d1ee31884f79398da59c99',
                'mmmu_random_fallback':'official parser, per-ID SHA256 seed; strict score excludes random fallback',
                'official_scnet_validation':False, 'baseline_tolerance_pass':None}
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Resume manifest mismatch'
    else:
        dump(manifest_path, manifest)
    records = {}
    raw_path = args.output/'predictions.jsonl'
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            record = json.loads(line)
            assert record['id'] not in records
            records[record['id']] = record
    pending = [r for r in samples if r['_id'] not in records]
    with raw_path.open('a', encoding='utf-8') as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for record in pool.map(lambda row: infer(row,args), pending):
            f.write(json.dumps(record, ensure_ascii=False)+'\n'); f.flush()
            records[record['id']] = record
            print(f"{args.dataset} {len(records)}/{len(samples)} {record['id']} error={record['error']}", flush=True)
    graded = [grade(r,records[r['_id']],args.dataset) for r in samples]
    with (args.output/'scored.jsonl').open('w',encoding='utf-8') as f:
        for record in graded:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    categories = defaultdict(lambda:{'count':0,'correct':0,'strict_correct':0})
    for r in graded:
        group=categories[r['category']]
        group['count']+=1; group['correct']+=int(r['correct']); group['strict_correct']+=int(r['strict_correct'])
    summary = {'dataset':args.dataset, 'created_at':datetime.now(timezone.utc).isoformat(),
               'count':len(graded), 'full_dataset':args.limit==0,
               'correct':sum(r['correct'] for r in graded),
               'strict_correct':sum(r['strict_correct'] for r in graded),
               'errors':sum(bool(r['error']) for r in graded),
               'truncated':sum(r.get('finish_reason')=='length' for r in graded),
               'random_fallbacks':sum(r['random_fallback'] for r in graded),
               'categories':dict(categories), 'official_scnet_validation':False,
               'baseline_tolerance_pass':None}
    summary['accuracy']=summary['correct']/len(graded)
    summary['strict_accuracy']=summary['strict_correct']/len(graded)
    summary['complete']=summary['errors']==0
    dump(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if summary['errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
