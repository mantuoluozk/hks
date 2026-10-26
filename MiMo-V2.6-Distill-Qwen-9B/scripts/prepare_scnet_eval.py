"""Rebuild the fixed public samples and verify byte identity before evaluation."""
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'gsm8k': '03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546',
            'mmlu': '84f929be5a22b7f7895961ef19f3842dc3d9a276a42b82c42a6fe95e1395682d'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sources = json.loads((ROOT / 'sources/evaluation-files.json').read_text())['files']
    for item in sources:
        path = ROOT / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
    dest = ROOT / 'cache/datasets/mmlu'
    with tarfile.open(dest / 'data.tar') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('_test.csv'):
                continue
            target = (dest / member.name).resolve()
            assert target.is_relative_to(dest.resolve()), member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
    subprocess.run([sys.executable, 'scripts/prepare_data.py'], cwd=ROOT, check=True)
    samples = {}
    for dataset, expected in EXPECTED.items():
        path = ROOT / 'data' / (dataset + '.jsonl')
        actual = sha(path)
        assert actual == expected, (dataset, actual, expected)
        count = sum(bool(line.strip()) for line in path.read_text().splitlines())
        assert count == 200
        samples[dataset] = {'count': count, 'sha256': actual}
    report = {'scope': 'fixed public self-evaluation; not official subset or publisher GPU baseline',
              'sample_rebuild_byte_identical': True, 'seed': 20261023,
              'sources': sources, 'samples': samples,
              'prepare_script_sha256': sha(ROOT / 'scripts/prepare_data.py')}
    out = ROOT / 'results/scnet/data-verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
