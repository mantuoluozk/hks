"""Check the image's installed dependency files against the pinned wheels."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
target = Path('/opt/qwen3-asr-deps')
assert target.is_dir(), 'Baked dependency directory is absent'
records = []
for item in json.loads((root/'sources/scnet-wheel-files.json').read_text()):
    wheel = root/'cache/wheels'/item['file']
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == item['sha256'], wheel
    assert md.version(item['package']) == item['version'], item['package']
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            path = target/name
            assert path.resolve().is_relative_to(target), name
            assert path.is_file(), name
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(archive.read(name)).digest(), name
            count += 1
    records.append({'package': item['package'], 'version': item['version'],
                    'wheel_sha256': item['sha256'], 'matched_files': count})
result = {'dependency_directory': str(target), 'all_files_match_pinned_wheels': True,
          'packages': records, 'verification': 'Wheel payload files, excluding installer-generated RECORD and bytecode.'}
out = Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True, exist_ok=True)
(out/'image-dependencies.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
