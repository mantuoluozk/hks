"""Download public OCRBench test and MMMU validation directly on the server.

Every Parquet file is verified against the upstream LFS SHA256 stored in the
snapshot manifest. Never downloads test answers from a model service.
"""
import concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'sources/datasets-metadata.json').read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def fetch(task):
    dataset,item=task
    dest=ROOT/'data/public'/dataset['repo']/item['path']
    dest.parent.mkdir(parents=True,exist_ok=True)
    expected=item['lfs']['oid']
    if dest.exists() and dest.stat().st_size==item['size'] and sha(dest)==expected:
        return str(dest)
    url=f"https://hf-mirror.com/datasets/{dataset['repo']}/resolve/main/{quote(item['path'])}"
    temp=dest.with_suffix(dest.suffix+'.download')
    log=ROOT/'logs'/('data-download-'+dataset['repo'].replace('/','-')+'-'+item['path'].replace('/','-')+'.log')
    for attempt in range(4):
        with log.open('a') as out:
            result=subprocess.run(['curl','-4','-fL','--connect-timeout','10','--max-time','240',
                '--retry','2','--retry-all-errors','--retry-delay','2','-o',str(temp),url],stdout=out,stderr=out)
        if result.returncode==0 and temp.stat().st_size==item['size'] and sha(temp)==expected:
            temp.replace(dest)
            print('Verified',dataset['repo'],item['path'],item['size'],flush=True)
            return str(dest)
        time.sleep(3)
    raise RuntimeError(f'Dataset download failed: {url}; see {log}')


tasks=[(dataset,item) for dataset in manifest for item in dataset['files']]
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    files=list(pool.map(fetch,tasks))
(ROOT/'results/public-data-download.json').write_text(json.dumps({'files':files,'source_manifest':'sources/datasets-metadata.json','all_sha256_verified':True},indent=2))
print('All public evaluation data downloaded and SHA256 verified.',flush=True)
