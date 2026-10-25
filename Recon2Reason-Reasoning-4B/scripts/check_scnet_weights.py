"""Validate SCNet-cloned files against our unchanged upstream SHA256 manifest."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('path', type=Path)
parser.add_argument('--source', choices=['scnet', 'modelscope'], default='scnet')
args = parser.parse_args()
required = {'config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.json',
            'tokenizer_config.json', 'preprocessor_config.json', 'special_tokens_map.json',
            'added_tokens.json', 'merges.txt', 'vocab.json', 'model.safetensors.index.json',
            'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'}
records = []
for line in (root / 'source-SHA256SUMS').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    name = name.removeprefix('./')
    if name not in required:
        continue
    path = args.path / name
    digest = hashlib.sha256()
    if path.is_file():
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
    else:
        actual = None
    records.append({'file': name, 'bytes': path.stat().st_size if path.is_file() else None,
                    'expected_sha256': expected, 'actual_sha256': actual, 'match': actual == expected})
assert {r['file'] for r in records} == required, 'Required file absent from upstream manifest'
record = {'created_at': datetime.now(timezone.utc).isoformat(),
          'source': 'SCNet AI community clone, reported by user' if args.source == 'scnet' else 'ModelScope BAAI upstream download on SCNet Notebook',
          'source_url': 'https://www.scnet.cn/ui/aihub/models/sugon_scnet/Recon2Reason-Reasoning-4B' if args.source == 'scnet' else 'https://modelscope.cn/models/BAAI/Recon2Reason-Reasoning-4B',
          'scnet_weight_source_requirement_met': args.source == 'scnet',
          'files': records, 'all_required_files_match': all(r['match'] for r in records),
          'official_acceptance_passed': None}
out = root / 'results/scnet/weights-check.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
assert record['all_required_files_match'], 'SCNet files differ; inspect before starting model'
