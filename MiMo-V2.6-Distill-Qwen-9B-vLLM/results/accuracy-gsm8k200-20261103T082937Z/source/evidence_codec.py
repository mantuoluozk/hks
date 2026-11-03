"""Lossless representation of consecutive cyclic benchmark token IDs."""
import hashlib,json

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
