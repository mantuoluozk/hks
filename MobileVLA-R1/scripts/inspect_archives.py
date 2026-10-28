"""Verify existing split archives and inspect ZIP entries without concatenation."""
import argparse,bisect,hashlib,io,json,os,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
class Parts(io.RawIOBase):
    def __init__(self,paths):
        super().__init__();self.paths=paths;self.offsets=[0];self.pos=0
        for p in paths:self.offsets.append(self.offsets[-1]+p.stat().st_size)
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        pos=offset+(self.pos if whence==1 else self.offsets[-1] if whence==2 else 0)
        if pos<0:raise ValueError('negative position')
        self.pos=pos;return pos
    def read(self,n=-1):
        n=self.offsets[-1]-self.pos if n<0 else min(n,self.offsets[-1]-self.pos)
        blocks=[]
        while n>0:
            index=bisect.bisect_right(self.offsets,self.pos)-1
            size=min(n,self.offsets[index+1]-self.pos)
            with self.paths[index].open('rb') as stream:
                stream.seek(self.pos-self.offsets[index]);raw=stream.read(size)
            if not raw:raise EOFError('short split archive')
            blocks.append(raw);self.pos+=len(raw);n-=len(raw)
        return b''.join(blocks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--path',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    meta=json.loads((ROOT/'sources/archive-manifest.json').read_text());entries=sorted(meta['files'],key=lambda x:x['path'])
    paths=[a.path/x['path'] for x in entries];checks=[]
    for entry,path in zip(entries,paths):
        assert path.is_file() and path.stat().st_size==entry['bytes'],path
        if a.verify:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            row={'path':entry['path'],'bytes':path.stat().st_size,'sha256':h.hexdigest(),'passed':h.hexdigest()==entry['sha256']}
            checks.append(row);print(json.dumps(row),flush=True)
            assert row['passed'],path
    if a.verify:(ROOT/'results/archive-check.json').write_text(json.dumps({'revision':meta['revision'],'files':checks,'passed':len(checks)==26 and all(x['passed'] for x in checks)},indent=2)+'\n')
    with zipfile.ZipFile(Parts(paths)) as archive:
        inventory=[];small={}
        for entry in archive.infolist():
            name=PurePosixPath(entry.filename)
            safe=not name.is_absolute() and '..' not in name.parts and '\\' not in entry.filename
            assert safe,entry.filename
            inventory.append({'path':entry.filename,'bytes':entry.file_size,'compressed_bytes':entry.compress_size,'encrypted':bool(entry.flag_bits&1)})
            if entry.file_size<300000 and not entry.is_dir() and name.name in ['config.json','generation_config.json','tokenizer_config.json','preprocessor_config.json','model_index.json','README.md','requirements.txt','pyproject.toml']:
                small[entry.filename]=archive.read(entry).decode('utf-8',errors='replace')
        (ROOT/'results/archive-inventory.json').write_text(json.dumps({'files':inventory,'total_uncompressed_bytes':sum(x['bytes'] for x in inventory),'small_configs':small,'weights_extracted':False},indent=2)+'\n')
        print(json.dumps({'zip_entries':len(inventory),'uncompressed_bytes':sum(x['bytes'] for x in inventory),'configs':list(small),'weights_extracted':False}),flush=True)

if __name__=='__main__':main()
