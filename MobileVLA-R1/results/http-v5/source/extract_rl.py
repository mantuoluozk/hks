"""Extract only the RL checkpoint after full split archive verification."""
import argparse,hashlib,json,shutil,stat,zipfile
from pathlib import Path,PurePosixPath
from inspect_archives import ROOT,Parts
p=argparse.ArgumentParser();p.add_argument('--path',type=Path,required=True);a=p.parse_args()
verified=json.loads((ROOT/'results/archive-check.json').read_text())
meta=json.loads((ROOT/'sources/archive-manifest.json').read_text())
assert verified['passed'] and verified['revision']==meta['revision']
parts=[a.path/x['path'] for x in sorted(meta['files'],key=lambda x:x['path'])]
target=ROOT/'weights/mobilevla-r1-rl';target.mkdir(parents=True,exist_ok=True)
records=[]
with zipfile.ZipFile(Parts(parts)) as archive:
    for entry in archive.infolist():
        if not entry.filename.startswith('weight/rl/') or entry.is_dir():continue
        relative=PurePosixPath(entry.filename).relative_to('weight/rl')
        assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in str(relative)
        assert not stat.S_ISLNK(entry.external_attr>>16)
        path=target/str(relative);assert path.resolve().is_relative_to(target.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        assert not path.exists(),str(path)
        partial=path.with_name(path.name+'.partial');h=hashlib.sha256();size=0
        with archive.open(entry) as source,partial.open('xb') as dest:
            for block in iter(lambda:source.read(8*1024*1024),b''):
                dest.write(block);h.update(block);size+=len(block)
        assert size==entry.file_size
        partial.rename(path)
        row={'path':str(relative),'bytes':size,'sha256':h.hexdigest()}
        records.append(row);print(json.dumps(row),flush=True)
result={'variant':'rl','model_loaded':False,'archive_crc_verified':True,
        'publisher_archive_revision':meta['revision'],'files':records,'bytes':sum(x['bytes'] for x in records)}
(ROOT/'results/extracted-rl.json').write_text(json.dumps(result,indent=2)+'\n')
