"""Public test-clean WER/CER self-evaluation; never mark official acceptance."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import base64
import editdistance
import requests
import soundfile as sf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'sources/whisper/whisper'))
from normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
prepared = json.loads((root/'results/dataset-preparation.json').read_text())
assert prepared['complete'] and prepared['utterances'] == 2620
items = []
for textfile in sorted((root/'data/LibriSpeech/test-clean').rglob('*.trans.txt')):
    for line in textfile.read_text().splitlines():
        uid, text = line.split(' ', 1)
        audio = textfile.parent / (uid+'.flac')
        assert audio.is_file()
        items.append((uid, audio, text))
assert len(items) == 2620
items.sort()
limit = int(os.environ.get('EVAL_LIMIT', '2620'))
assert 1 <= limit <= 2620
items = items[:limit]
out = Path(os.environ.get('EVAL_OUTPUT_DIR',str(root/'results'/('librispeech-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))))
out.mkdir(parents=True)
runtime = json.loads((Path(os.environ.get('RUN_RESULTS_DIR',str(root/'results')))/'runtime-config.json').read_text())
base = f'http://127.0.0.1:{runtime["PORT"]}'
workers = int(os.environ.get('EVAL_WORKERS', '4'))

def evaluate(item):
    uid, path, reference = item
    expected = normalizer(reference)
    session = requests.Session();session.trust_env = False
    data_url = 'data:audio/flac;base64,' + base64.b64encode(path.read_bytes()).decode()
    t0 = time.perf_counter()
    r = session.post(base+'/v1/chat/completions', json={
        'model': runtime['SERVED_NAME'], 'messages': [{'role': 'user', 'content': [
            {'type': 'audio_url', 'audio_url': {'url': data_url}}]}],
        'temperature': 0, 'max_tokens': 1024}, timeout=300)
    r.raise_for_status();response = r.json()
    choice = response['choices'][0]
    raw = choice['message']['content']
    hypothesis = normalizer(raw.split('<asr_text>', 1)[-1])
    return {'id': uid, 'hypothesis': hypothesis, 'raw_output': raw,
            'reference_normalized_sha256': hashlib.sha256(expected.encode()).hexdigest(),
            'reference_words': len(expected.split()),
            'word_errors': editdistance.eval(expected.split(), hypothesis.split()),
            'reference_chars': len(expected.replace(' ', '')),
            'char_errors': editdistance.eval(expected.replace(' ', ''), hypothesis.replace(' ', '')),
            'truncated': choice['finish_reason'] == 'length', 'finish_reason': choice['finish_reason'],
            'elapsed_seconds': time.perf_counter()-t0, 'audio_seconds': sf.info(path).duration}

rows = [];failures = [];t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=workers) as pool, (out/'predictions.jsonl').open('w', encoding='utf-8') as f:
    futures = {pool.submit(evaluate, item): item[0] for item in items}
    for future in cf.as_completed(futures):
        uid = futures[future]
        try:
            row = future.result();rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False)+'\n');f.flush()
        except Exception as e:
            failures.append({'id': uid, 'error_type': type(e).__name__, 'error': str(e)})
        if (len(rows)+len(failures)) % 100 == 0:print('Evaluated', len(rows), 'failures', len(failures), flush=True)
words = sum(r['reference_words'] for r in rows)
chars = sum(r['reference_chars'] for r in rows)
report = {'dataset': 'LibriSpeech test-clean', 'dataset_sha256': prepared.get('source_sha256', prepared.get('archive_sha256')),
          'requested': len(items), 'completed': len(rows), 'failures': failures,
          'complete': len(rows) == len(items) and not failures,
          'wer_percent': 100*sum(r['word_errors'] for r in rows)/words if words else None,
          'cer_percent': 100*sum(r['char_errors'] for r in rows)/chars if chars else None,
          'truncated': sum(r['truncated'] for r in rows), 'workers': workers,
          'elapsed_seconds': time.perf_counter()-t0, 'runtime': runtime,
          'normalization': 'OpenAI Whisper EnglishTextNormalizer; CER excludes spaces after the same normalization.',
          'normalizer_commit': '86098128c0b4f24f0e2aa2994de830614b474227',
          'selection': 'All 2620 test-clean utterances' if limit == 2620 else 'Lexicographically first utterance IDs; diagnostic subset only',
          'publisher_test_clean_wer_percent': 1.63,
          'publisher_normalization_equivalence_verified': False,
          'official_accuracy_pass': None, 'official_scnet_validation': False}
(out/'summary.json').write_text(json.dumps(report, indent=2))
(root/'results/accuracy-latest.txt').write_text(str(out))
print(json.dumps(report, indent=2))
assert report['complete'] and report['truncated'] == 0, 'Incomplete or truncated public self-evaluation'
