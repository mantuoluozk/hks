"""Pinned benchmark prompt/extraction helpers, without loading an inference model."""
import ast
import contextlib
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/mmlu-pro-reference/evaluate_from_api.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

@lru_cache(maxsize=1)
def upstream():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    names={'preprocess','format_example','extract_answer','extract_again','extract_final'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==len(names)
    namespace={'re':re}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),namespace)
    single=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='single_request')
    header=next(n.value for n in single.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in n.targets))
    return namespace,header,single

def prompt(item,examples):
    namespace,header,_=upstream()
    category=item['category']
    assert len(examples)==5, 'Exactly five validation examples per category are required'
    text=eval(compile(ast.Expression(body=header),'upstream_prompt_header','eval'),{'category':category})
    for example in examples:
        assert example['category']==category
        text+=namespace['format_example'](example['question'],example['options'],example['cot_content'])
    return text+namespace['format_example'](item['question'],item['options'])

def parse_answer(content,option_count):
    namespace,_,_=upstream()
    with contextlib.redirect_stdout(io.StringIO()):
        answer=namespace['extract_answer']((content or '').replace('**',''))
    return answer if isinstance(answer,str) and len(answer)==1 and answer in 'ABCDEFGHIJ'[:option_count] else None

def score_choice(raw,expected,option_count):
    choice=raw['choices'][0]
    content=choice.get('message',{}).get('content') or ''
    predicted=parse_answer(content,option_count) if content.strip() else None
    truncated=choice.get('finish_reason')=='length'
    return {'predicted':predicted,'correct':predicted==expected and not truncated,
            'unparsed':predicted is None,'truncated':truncated,'finish_reason':choice.get('finish_reason')}

def assert_auk_idle():
    path=ROOT.parent/'AuK/results/seedtts-full-seed1234/state.json'
    if path.exists():
        phase=json.loads(path.read_text())['phase']
        if phase in {'server_startup','generating','transcribing'}:
            raise RuntimeError(f'AuK is still {phase}; do not start MiMo evaluation on this Notebook')

def aggregate(rows,expected_ids,reference,full):
    ids=[r['id'] for r in rows]
    assert len(ids)==len(set(ids)), 'Duplicate predictions'
    assert set(ids)<=set(expected_ids), 'Unexpected question IDs'
    total=len(expected_ids)
    correct=sum(bool(r.get('correct')) for r in rows)
    count=100*correct/total if total else 0.0
    categories={}
    for row in rows:
        group=categories.setdefault(row['category'],{'total':0,'correct':0})
        group['total']+=1;group['correct']+=bool(row.get('correct'))
    for group in categories.values():group['accuracy_percent']=100*group['correct']/group['total']
    complete=set(ids)==set(expected_ids)
    errors=sum(bool(r.get('error')) for r in rows)
    return {'completed':len(rows),'expected':total,'missing':total-len(rows),'correct':correct,
            'accuracy_percent':count,'full_test_split':full,'complete':complete,'errors':errors,
            'truncated':sum(bool(r.get('truncated')) for r in rows),
            'unparsed':sum(bool(r.get('unparsed')) for r in rows),'by_category':categories,
            'reference_accuracy_percent':reference,
            'numeric_threshold_met':count>=reference if full and complete and errors==0 else None,
            'publisher_exact_protocol_confirmed':False,'official_acceptance_claimed':False}
