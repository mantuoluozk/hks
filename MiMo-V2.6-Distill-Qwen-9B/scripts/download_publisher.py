"""Fetch the fixed publisher snapshot, or verify files supplied offline."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO='XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B'
REVISION='2367e865d009c13ac81713a2878291d33ab28177'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dest',type=Path,default=Path('/root/private_data/hks/weights/MiMo-V2.6-Distill-Qwen-9B'))
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();args.dest.mkdir(parents=True,exist_ok=True)
    files=json.loads((ROOT/'scripts/files.json').read_text())['Data']['Files']
    records=[]
    for item in files:
        name=item['Name']
        if name in {'README.md','.gitattributes','configuration.json'}:continue
        assert Path(name).name==name
        path=args.dest/name
        if not path.exists() and not args.verify_only:
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPO,name,revision=REVISION,local_dir=str(args.dest))
        assert path.is_file() and path.stat().st_size==item['Size'],f'Missing/incomplete: {name}'
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(16*1024**2),b''):h.update(chunk)
        assert h.hexdigest()==item['Sha256'],f'Checksum mismatch: {name}'
        records.append({'file':name,'bytes':item['Size'],'sha256':h.hexdigest()})
    assert len(records)==15
    (args.dest/'upstream-source.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,
        'source':'publisher Hugging Face snapshot','files':records},indent=2))
    print('Verified 15 runtime files; source manifest written')


if __name__=='__main__':main()
