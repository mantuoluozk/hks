"""Verified ModelScope mirror of the complete public test-clean Parquet file."""
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode
import requests
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1]
folder = root/'data/librispeech-source';folder.mkdir(parents=True, exist_ok=True)
size = 350452636
expected = '7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0'
revision = '2a4a7918cd87e2f0e315e576b405ae7b7a65d83f'
source = 'https://modelscope.cn/api/v1/datasets/openslr/librispeech_asr/repo?' + urlencode({
    'Revision': revision, 'FilePath': 'clean/test/0000.parquet', 'Source': 'SDK'})
archive = folder/'test-clean.parquet'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''):h.update(b)
    return h.hexdigest()

if not archive.exists():
    with requests.get(source, headers={'Range':'bytes=0-0'}, stream=True, timeout=(20,60)) as r:
        r.raise_for_status();assert r.status_code == 206;cdn = r.url
    parts = folder/'parts';parts.mkdir(exist_ok=True)
    chunk = 4 << 20
    ranges = [(i,min(i+chunk,size)-1) for i in range(0,size,chunk)]
    def fetch(pair):
        start,end = pair;p = parts/str(start)
        if p.exists() and p.stat().st_size == end-start+1:return p
        for attempt in range(8):
            try:
                with requests.get(cdn, headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=(20,60)) as r:
                    r.raise_for_status()
                    assert r.status_code == 206 and r.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    incoming = p.with_suffix('.incoming')
                    with incoming.open('wb') as f:
                        for b in r.iter_content(1 << 20):
                            f.write(b);assert f.tell() <= end-start+1
                assert incoming.stat().st_size == end-start+1
                incoming.replace(p);print('Dataset bytes', end+1, '/', size, flush=True);return p
            except Exception as e:
                print('Dataset chunk retry', start, type(e).__name__, flush=True)
                if attempt == 7:raise
                time.sleep(3)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:files = list(pool.map(fetch,ranges))
    tmp = archive.with_suffix('.incoming')
    with tmp.open('wb') as f:
        for p in files:
            with p.open('rb') as source_file:
                for b in iter(lambda:source_file.read(1 << 20), b''):f.write(b)
    assert tmp.stat().st_size == size and sha(tmp) == expected
    tmp.replace(archive)
assert archive.stat().st_size == size and sha(archive) == expected
chapters = {};ids = set()
for batch in pq.ParquetFile(archive).iter_batches(batch_size=128, columns=['id','text','audio']):
    for row in batch.to_pylist():
        uid = row['id'];speaker,chapter,_ = uid.split('-');assert uid not in ids
        assert all(x.isdigit() for x in uid.split('-'))
        ids.add(uid)
        dest = root/'data/LibriSpeech/test-clean'/speaker/chapter;dest.mkdir(parents=True,exist_ok=True)
        audio = row['audio']['bytes'];assert audio[:4] == b'fLaC'
        (dest/(uid+'.flac')).write_bytes(audio)
        chapters.setdefault((dest,speaker,chapter),[]).append(uid+' '+row['text'])
assert len(ids) == 2620
for (dest,speaker,chapter),lines in chapters.items():
    (dest/f'{speaker}-{chapter}.trans.txt').write_text('\n'.join(sorted(lines))+'\n')
report = {'dataset':'LibriSpeech test-clean','utterances':2620,'complete':True,
          'source_format':'Parquet mirror, extracted without audio resampling',
          'source_sha256':expected,'source_bytes':size,'revision':revision,
          'source_url':source,'upstream':'https://huggingface.co/datasets/openslr/librispeech_asr/blob/main/clean/test/0000.parquet',
          'official_competition_subset':False}
(root/'results/dataset-preparation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
