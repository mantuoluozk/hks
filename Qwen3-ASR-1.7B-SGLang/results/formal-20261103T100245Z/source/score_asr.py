"""Fixed Whisper normalization and explicit word substitution/deletion/insertion."""
import hashlib,json,sys
from pathlib import Path
import editdistance
ROOT=Path(__file__).resolve().parents[1]
for f in json.loads((ROOT/'sources/normalizer-files.json').read_text())['files']:
    assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256']
sys.path.insert(0,str(ROOT/'sources/whisper/whisper'))
from normalizers import EnglishTextNormalizer
normalize=EnglishTextNormalizer()
def word_counts(reference,hypothesis):
    a=reference.split();b=hypothesis.split();n=len(a);m=len(b)
    dp=[list(range(m+1))]+[[i]+[0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]!=b[j-1]))
    i=n;j=m;s=d=ins=0
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(a[i-1]!=b[j-1]):s+=a[i-1]!=b[j-1];i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:d+=1;i-=1
        else:ins+=1;j-=1
    assert s+d+ins==editdistance.eval(a,b)
    return {'substitutions':s,'deletions':d,'insertions':ins,'reference_words':n,'word_errors':s+d+ins}
def score(reference,text):
    expected=normalize(reference);hyp=normalize(text);counts=word_counts(expected,hyp)
    return {**counts,'hypothesis':hyp,'reference_normalized_sha256':hashlib.sha256(expected.encode()).hexdigest(),
            'reference_chars':len(expected.replace(' ','')),'char_errors':editdistance.eval(expected.replace(' ',''),hyp.replace(' ',''))}
def summary(rows):
    assert len(rows)==2620 and len({r['id'] for r in rows})==2620
    words=sum(r['reference_words'] for r in rows);chars=sum(r['reference_chars'] for r in rows)
    errors=sum(not r['success'] for r in rows);truncated=sum(r.get('finish_reason')=='length' for r in rows)
    return {'complete':errors==0 and truncated==0,'requested':2620,'completed':len(rows),'successful':2620-errors,'request_errors':errors,'truncated':truncated,
            'reference_words':words,'word_errors':sum(r['word_errors'] for r in rows),'substitutions':sum(r['substitutions'] for r in rows),'deletions':sum(r['deletions'] for r in rows),'insertions':sum(r['insertions'] for r in rows),
            'reference_chars':chars,'char_errors':sum(r['char_errors'] for r in rows),'wer_percent':100*sum(r['word_errors'] for r in rows)/words,'cer_percent':100*sum(r['char_errors'] for r in rows)/chars,
            'dataset':'LibriSpeech test-clean full 2620','workers':4,'normalizer_commit':'86098128c0b4f24f0e2aa2994de830614b474227','publisher_reference_wer_percent':1.63,'publisher_normalization_equivalence_verified':False,'official_acceptance':None}
