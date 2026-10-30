"""Convert an already downloaded OCRBench parquet; never download dataset files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('parquet', type=Path)
p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/OCRBench')
a = p.parse_args()
expected = 'd9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5'
assert hashlib.sha256(a.parquet.read_bytes()).hexdigest() == expected
rows = pq.read_table(a.parquet).to_pylist()
assert len(rows) == 1000
(a.output / 'img').mkdir(parents=True, exist_ok=True)
records = []
for i, row in enumerate(rows):
    image = row['image']
    if image.get('bytes') is not None:
        im = Image.open(io.BytesIO(image['bytes']))
    else:
        im = Image.open(a.parquet.parent / image['path'])
    im.convert('RGB').save(a.output / 'img' / f'{i}.png')
    records.append({'question_id': str(i), 'question': row['question'], 'answer': row['answer'],
                    'question_type': row['question_type'], 'dataset': row.get('dataset', row.get('dataset_name')),
                    'img_path': f'img/{i}.png'})
assert all(r['dataset'] is not None for r in records)
(a.output / 'data.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Prepared 1000 OCRBench questions; dataset stays outside the Git export')
