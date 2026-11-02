"""Download and verify the fixed publisher ChartQA subset; extract regular files safely."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / 'sources/evaluation-source.json').read_text())
    dest = ROOT / 'data/Trident-Eval'
    records = []
    for entry in source['files']:
        path = dest / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://hf-mirror.com/datasets/{source['repository']}/resolve/{source['revision']}/{entry['path']}"
            temp = path.with_name(path.name + '.partial')
            with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as output:
                while chunk := response.read(1 << 20):
                    output.write(chunk)
            assert temp.stat().st_size == entry['size'], entry['path']
            temp.replace(path)
        data = path.read_bytes()
        assert len(data) == entry['size'], entry['path']
        digest = hashlib.sha256(data).hexdigest()
        if entry.get('lfs'):
            assert digest == entry['lfs']['oid'], entry['path']
        else:
            git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert git_blob == entry['oid'], entry['path']
        records.append({'path':entry['path'], 'bytes':len(data), 'sha256':digest, 'publisher_match':True})
        print('VERIFIED', entry['path'], flush=True)
    target = dest / 'ChartQA'
    with tarfile.open(target / 'images.tar.gz') as archive:
        for member in archive.getmembers():
            assert (member.isfile() or member.isdir()) and (target / member.name).resolve().is_relative_to(target.resolve()), member.name
        archive.extractall(target)
    out = Path(os.environ.get('RUN_RESULTS_DIR', ROOT / 'results/dataset'))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'chartqa-check.json').write_text(json.dumps({'repository':source['repository'], 'revision':source['revision'], 'files':records, 'all_verified':True}, indent=2))
    print('ChartQA ready:', target, flush=True)


if __name__ == '__main__':
    main()
