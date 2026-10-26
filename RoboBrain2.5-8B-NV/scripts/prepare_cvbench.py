"""Download and verify the fixed public dataset and FlagEvalMM source snapshots.

Run inside the Notebook. For restricted networks use a network-enabled host to
stream the same pinned files into these cache paths; never change the hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    config = json.loads((ROOT/'configs/cvbench-protocol.json').read_text())
    endpoint = os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
    sources = []
    for item in config['upstream_files']:
        url = (f"https://raw.githubusercontent.com/flageval-baai/FlagEvalMM/"
               f"{config['upstream_revision']}/{item['path']}")
        sources.append((ROOT/'cache/flagevalmm'/item['path'],url,item['sha256']))
    for item in config['dataset_files']:
        url = (f"{endpoint}/datasets/{config['dataset_repository']}/resolve/"
               f"{config['dataset_revision']}/{item['file']}")
        sources.append((ROOT/'cache/datasets/CV-Bench'/item['file'],url,item['sha256']))
    for path,url,expected in sources:
        if path.exists() and digest(path)==expected:
            print('VERIFIED',str(path.relative_to(ROOT)),flush=True)
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_name(path.name+'.download')
        with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as f:
            for block in iter(lambda:response.read(1024*1024),b''):
                f.write(block)
        assert digest(temp)==expected, ('Checksum mismatch',path)
        temp.replace(path)
        print('DOWNLOADED_VERIFIED',str(path.relative_to(ROOT)),flush=True)


if __name__=='__main__':
    main()
