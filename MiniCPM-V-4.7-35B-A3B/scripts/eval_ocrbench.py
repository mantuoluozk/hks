"""OCRBench full-set evaluation against the local vLLM OpenAI server.

Dataset: publisher bundle at data/OCRBench (1000 items, HF echo840/OCRBench test,
parquet sha256 D9CC9F488E5310143D8D89956C19D0BDFD71E7089025DA5202643F73720DF7F5).
Protocol (declared, non-official): greedy, enable_thinking=False, 16x default
downsample, max_new_tokens 64, image sent as data URL. Scoring reuses the
publisher eval logic (FlagEvalMM tasks/ocrbench/evaluate.py, Apache-2.0).
"""
import base64, concurrent.futures, json, statistics, sys, time, unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/OCRBench'
OUT = ROOT / 'results/accuracy/ocrbench'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30147
BASE = f'http://127.0.0.1:{PORT}'
MODEL = 'minicpm-v-4.7'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = full 1000


def data_url(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def ask(item, retries=2):
    body = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': data_url(DATA / item['img_path'])}},
            {'type': 'text', 'text': item['question']},
        ]}],
        'temperature': 0, 'max_tokens': 64, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    req = urllib.request.Request(BASE + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    last = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                obj = json.loads(r.read().decode())
            return obj['choices'][0]['message'].get('content') or ''
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5)
    raise RuntimeError(f'request failed for {item["question_id"]}: {last}')


# --- publisher scoring (verbatim semantics from FlagEvalMM ocrbench evaluate.py) ---
def get_score(gt, pred):
    dataset_name = gt['dataset']
    answers = gt['answer'] if isinstance(gt['answer'], list) else [gt['answer']]

    def process(x):
        if dataset_name in ('HME100k', 'HMER'):
            return unicodedata.normalize('NFKC', x).strip().replace('\n', ' ').replace(' ', '')
        return x.lower().strip().replace('\n', ' ')

    processed = process(pred)
    return int(any(process(a) in processed for a in answers))


def main():
    items = json.loads((DATA / 'data.json').read_text(encoding='utf-8'))
    if LIMIT:
        items = items[:LIMIT]
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / 'predictions.jsonl'
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            done[r['question_id']] = r
    score_rows = []
    started = time.perf_counter()
    with results_path.open('a', encoding='utf-8') as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(ask, it): it for it in items if it['question_id'] not in done}
            for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                it = futs[fut]
                try:
                    text = fut.result()
                except RuntimeError as e:
                    print(f'FAIL {it["question_id"]}: {e}')
                    raise
                rec = {'question_id': it['question_id'], 'dataset': it['dataset'],
                       'question_type': it['question_type'], 'answer': it['answer'],
                       'prediction': text, 'score': get_score(it, text)}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                score_rows.append(rec)
                if n % 50 == 0:
                    el = time.perf_counter() - started
                    cur = sum(r['score'] for r in score_rows) / len(score_rows)
                    print(f'{n}/{len(futs)} elapsed={el:.0f}s cur_acc={cur * 100:.2f}', flush=True)
    rows = list(done.values()) + score_rows
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r['question_type']][0] += r['score']
        by_type[r['question_type']][1] += 1
    total = sum(v[0] for v in by_type.values())
    summary = {
        'final_score': [total, len(rows)],
        'accuracy_x100': round(total / len(rows) * 100, 4),
        'official_declared': 91.2,
        'protocol': 'greedy, enable_thinking=False, 16x, max_new_tokens=64, local protocol (non-official)',
        'model_revision': 'ee2d0a272c9d55db52a8b6841b6b24c9d0dda0a3',
        'dataset_sha256': 'D9CC9F488E5310143D8D89956C19D0BDFD71E7089025DA5202643F73720DF7F5',
        'per_type': {k: v for k, v in sorted(by_type.items())},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
