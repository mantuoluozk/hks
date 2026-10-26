"""Verify demo audio already provided locally; download only with an explicit flag."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
out=ROOT/'data/examples';out.mkdir(parents=True,exist_ok=True)
for item in json.loads((ROOT/'sources/audio-sources.json').read_text()):
    path=out/item['file']
    if not path.exists():
        assert args.download, f'Provide {path} from {item["url"]}; no automatic download requested'
        with urllib.request.urlopen(item['url'],timeout=120) as response:path.write_bytes(response.read())
    assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    print('Verified demo audio',path.name)
