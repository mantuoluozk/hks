"""Lossless representation of consecutive cyclic benchmark token IDs."""
import hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_text(path):
    """Transparent byte-exact decompression of compact publication file stubs."""
    blob=Path(path).read_bytes()
    if len(blob)<2048:
        try:ref=json.loads(blob)
        except (ValueError,UnicodeDecodeError):ref=None
        if isinstance(ref,dict) and ref.get('encoding')=='file-zlib-v1':
            parts=[]
            for item in ref['assets']:
                p=ROOT/item['path'];assert p.resolve().is_relative_to(ROOT)
                data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256'];parts.append(data)
            blob=zlib.decompress(b''.join(parts))
            assert len(blob)==ref['bytes'] and hashlib.sha256(blob).hexdigest()==ref['sha256']
    return blob.decode('utf-8')

def digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def encode(ids):
    if not isinstance(ids,list) or not ids or not all(type(x) is int and x>=0 for x in ids):return ids
    wraps=[ids[i-1]+1 for i in range(1,len(ids)) if ids[i]==0]
    modulus=wraps[0] if wraps else max(ids)+1
    if not all(x==(ids[0]+i)%modulus for i,x in enumerate(ids)):return ids
    result={'encoding':'cyclic-range-v1','start':ids[0],'count':len(ids),'modulus':modulus,'sha256':digest(ids)}
    assert decode(result)==ids
    return result

def decode(value):
    if isinstance(value,list):return value
    assert value['encoding']=='cyclic-range-v1'
    n=value['count'];m=value['modulus'];s=value['start']
    assert type(n) is int and 0<n<=32768 and type(m) is int and 0<m<=248320 and 0<=s<m
    ids=[(s+i)%m for i in range(n)]
    assert digest(ids)==value['sha256']
    return ids
