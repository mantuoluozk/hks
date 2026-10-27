"""Prepare small, pinned source/wheel dependencies without downloading any model."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
COMMIT='ea981b8b7d80f10deaff369bcda8cbf2c90428dd'


def download(url,path,expected=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response,path.with_suffix('.partial').open('wb') as output:
            size=0
            while block:=response.read(1<<20):
                size+=len(block)
                if size>64<<20:raise RuntimeError('Unexpectedly large dependency; review before continuing')
                output.write(block)
        path.with_suffix('.partial').replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected:assert digest==expected,path
    return {'url':url,'filename':path.name,'bytes':path.stat().st_size,'sha256':digest}


def main():
    entries=[]
    for name,version in [('transformers','5.17.0'),('tokenizers','0.23.1')]:
        metadata=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30))
        candidates=[x for x in metadata['urls'] if x['filename'].endswith('.whl') and
                    ('py3-none-any' in x['filename'] if name=='transformers' else 'cp310-abi3-manylinux_2_17_x86_64' in x['filename'])]
        assert len(candidates)==1
        item=candidates[0]
        entries.append(download(item['url'],ROOT/'runtime/wheels'/item['filename'],item['digests']['sha256']))
        print('Verified',item['filename'],flush=True)
    url=f'https://codeload.github.com/huggingface/diffusers/zip/{COMMIT}'
    archive=ROOT/'cache'/f'diffusers-{COMMIT}.zip'
    entry=download(url,archive);entry['revision']=COMMIT;entries.append(entry)
    dest=ROOT/'runtime/diffusers-py310'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            relative=Path(*Path(name).parts[1:])
            if not relative.parts or name.endswith('/'):continue
            if relative.parts[0]!='src' and str(relative) not in {'LICENSE','setup.py','pyproject.toml','README.md'}:continue
            target=dest/relative
            assert target.resolve().is_relative_to(dest.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    (ROOT/'sources/runtime-dependencies.json').write_text(json.dumps({'packages':entries,'core_framework_policy':'Keep existing vendor torch, torchvision and torchaudio. Install wheels --no-deps only into the model runtime directory.'},indent=2))
    print('Runtime prepared',sum(f.stat().st_size for f in (ROOT/'runtime').rglob('*') if f.is_file()),flush=True)


if __name__=='__main__':main()
