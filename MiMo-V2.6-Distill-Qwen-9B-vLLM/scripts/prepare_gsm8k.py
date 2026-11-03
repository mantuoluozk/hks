"""Validate an existing 419088-byte pinned dataset; no network or download."""
import hashlib,json,os,random
from pathlib import Path
import pyarrow.parquet as pq
from gsm8k_score import extract_answer,INVALID_ANS
ROOT=Path(__file__).resolve().parents[1]
default=ROOT/'data/gsm8k/test.parquet'
if not default.exists():default=ROOT.parent/'MiMo-V2.6-Distill-Qwen-9B/cache/datasets/gsm8k/main/test-00000-of-00001.parquet'
source=Path(os.environ.get('GSM8K_PARQUET',str(default)))
assert source.stat().st_size==419088
sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha=='ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59'
rows=pq.read_table(source).to_pylist();assert len(rows)==1319 and len({r['question'] for r in rows})==1319
items=[{'question':str(r['question']).strip(),'answer':str(r['answer']).split('####')[-1].strip().replace(',','')} for r in rows if '####' in str(r['answer'])]
random.Random(20261023).shuffle(items);rows=items[:200];assert len(rows)==200
out=ROOT/'data/gsm8k';out.mkdir(parents=True,exist_ok=True)
target=out/'test.jsonl';target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
assert hashlib.sha256(target.read_bytes()).hexdigest()=='03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546'
proof={'repository':'openai/gsm8k','revision':'740312add88f781978c0658806c59bc2815b9866','file':'main/test-00000-of-00001.parquet',
       'url':'https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet',
       'source_path':str(source),'source_bytes':419088,'source_sha256':sha,'source_test_count':1319,'test_count':200,'sample_seed':20261023,
       'sglang_sample_byte_identical':True,
       'prepared_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'prepared_bytes':target.stat().st_size}
(out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
