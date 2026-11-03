"""Use the unchanged published SGLang prompt and Decimal scoring functions."""
import ast,hashlib,json,re
from decimal import Decimal,InvalidOperation
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'sources/sglang-public-eval.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='19ea84fdf23729c8a12ee1072628b4df595e738ad04fb7347dbdba4d24aa5247'
tree=ast.parse(source.read_text());nodes=[]
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['number','score','prompt']:nodes.append(node)
assert len(nodes)==3
scope={'re':re,'Decimal':Decimal,'InvalidOperation':InvalidOperation};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
score=scope['score'];INVALID_ANS=None
def prompt(item):return scope['prompt']('gsm8k',item)
def extract_answer(content):return score('gsm8k',content,'0')[0]
def summarize(rows):
    assert len(rows)==200 and {r['index'] for r in rows}==set(range(200))
    for r in rows:
        content='' if r['error'] else (r['response']['choices'][0]['message'].get('content') or '')
        prediction,matched=score('gsm8k',content,r['expected'])
        correct=not r['error'] and matched
        assert r['predicted']==prediction and r['correct']==correct
    correct=sum(r['correct'] for r in rows);errors=sum(bool(r['error']) for r in rows)
    return {'complete':True,'dataset':'openai/gsm8k fixed SGLang-aligned sample','total':200,'correct':correct,
            'accuracy_percent':100*correct/200,'request_errors':errors,
            'truncated':sum(r['truncated'] for r in rows),'unparsed':sum(r['predicted']==INVALID_ANS for r in rows),
            'technical_checks_passed':errors==0,'official_accuracy_pass':None,'official_acceptance':None,
            'scoring':'Unchanged SGLang public_eval.py: strip bold; last #### numeric; Decimal equality; final content only'}
