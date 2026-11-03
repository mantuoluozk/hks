"""Execute pinned upstream GSM8K extraction/scoring functions without Torch."""
import ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'sources/gsm8k-reference/dataset.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='c22b81bccaaebfd1123be296e8b3c3dca3ddc704c46b70bf4a50603bb627e197'
tree=ast.parse(source.read_text());nodes=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['ANS_RE','INVALID_ANS'] for t in node.targets):nodes.append(node)
    if isinstance(node,ast.FunctionDef) and node.name in ['extract_answer','is_correct']:nodes.append(node)
assert len(nodes)==4
scope={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
extract_answer=scope['extract_answer'];is_correct=scope['is_correct'];INVALID_ANS=scope['INVALID_ANS']
INSTRUCTION='Solve the math problem. Explain briefly, then finish with #### followed by the final numeric answer.\n\n'
def prompt(item):return INSTRUCTION+item['question']
def summarize(rows):
    assert len(rows)==1319 and {r['index'] for r in rows}==set(range(1319))
    for r in rows:
        content='' if r['error'] else (r['response']['choices'][0]['message'].get('content') or '')
        prediction=extract_answer(content)
        correct=not r['error'] and prediction!=INVALID_ANS and prediction==r['expected']
        assert r['predicted']==prediction and r['correct']==correct
    correct=sum(r['correct'] for r in rows);errors=sum(bool(r['error']) for r in rows)
    return {'complete':True,'dataset':'openai/gsm8k main test','total':1319,'correct':correct,
            'accuracy_percent':100*correct/1319,'request_errors':errors,
            'truncated':sum(r['truncated'] for r in rows),'unparsed':sum(r['predicted']==INVALID_ANS for r in rows),
            'technical_checks_passed':errors==0,'official_accuracy_pass':None,'official_acceptance':None,
            'scoring':'Pinned upstream first #### numeric extraction; comma removal then exact string match; final content only'}
