"""CPU fixtures only: protocol parity, fair scoring, transport failures and resume."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
import unittest
from unittest.mock import patch

import mmlu_pro_common as common
from mmlu_pro_common import aggregate,parse_answer,prompt,score_choice,upstream,read_jsonl
from mmlu_pro_eval import evaluate,request_body,http_health

def raw(content,reasoning=None,finish='stop'):
    return {'choices':[{'message':{'content':content,'reasoning_content':reasoning},'finish_reason':finish}]}

class ProtocolTests(unittest.TestCase):
    def test_unicode_line_separator_is_inside_json_string(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rows.jsonl'
            path.write_text(json.dumps({'question':'first\u2028second'},ensure_ascii=False)+'\n',encoding='utf-8')
            self.assertEqual(read_jsonl(path),[{'question':'first\u2028second'}])
    def test_prompt_matches_actual_upstream_call(self):
        item={'question_id':100,'category':'fixture','question':'Which option?','options':['first','second'],'answer':'B'}
        examples=[dict(item,question_id=i,question=f'Example {i}',cot_content='A: explanation. The answer is (A).') for i in range(5)]
        namespace,_,single=upstream();namespace=dict(namespace)
        captured=[]
        namespace['call_api']=lambda client,instruction,inputs:captured.append(instruction+inputs) or 'The answer is (B).'
        exec(compile(ast.Module(body=[single],type_ignores=[]),'upstream_single_request','exec'),namespace)
        namespace['single_request'](None,item,{'fixture':examples},[])
        self.assertEqual(prompt(item,examples),captured[0])
        self.assertEqual(prompt(dict(item,answer='A'),examples),captured[0])
    def test_requires_five_examples(self):
        with self.assertRaises(AssertionError):prompt({'category':'fixture'},[])
    def test_canonical_answer(self):self.assertEqual(parse_answer('The answer is (J).',10),'J')
    def test_markdown_cleaning(self):self.assertEqual(parse_answer('The answer is **(B)**.',4),'B')
    def test_no_guess_on_unparseable(self):self.assertIsNone(parse_answer('No valid response.',10))
    def test_out_of_range(self):self.assertIsNone(parse_answer('The answer is (J).',4))
    def test_never_parse_hidden_reasoning(self):
        self.assertTrue(score_choice(raw('The answer is (B).','The answer is (A).'),'B',4)['correct'])
        self.assertFalse(score_choice(raw(None,'The answer is (B).'),'B',4)['correct'])
    def test_truncation_is_wrong_even_if_letter_present(self):
        row=score_choice(raw('The answer is (A).',finish='length'),'A',4)
        self.assertTrue(row['truncated']);self.assertFalse(row['correct'])
    def test_incomplete_denominator(self):
        s=aggregate([{'id':1,'category':'fixture','correct':True}],[1,2],82.5,True)
        self.assertEqual(s['accuracy_percent'],50);self.assertFalse(s['complete']);self.assertIsNone(s['numeric_threshold_met'])
    def test_no_two_percent_tolerance(self):
        rows=[{'id':i,'category':'fixture','correct':i<824} for i in range(1000)]
        self.assertFalse(aggregate(rows,list(range(1000)),82.5,True)['numeric_threshold_met'])
    def test_equal_threshold_is_pass(self):
        rows=[{'id':i,'category':'fixture','correct':i<33} for i in range(40)]
        self.assertTrue(aggregate(rows,list(range(40)),82.5,True)['numeric_threshold_met'])
    def test_pilot_never_reports_full_threshold(self):
        self.assertIsNone(aggregate([{'id':1,'category':'fixture','correct':True}],[1],82.5,False)['numeric_threshold_met'])
    def test_duplicates_rejected(self):
        with self.assertRaises(AssertionError):aggregate([{'id':1},{'id':1}],[1],82.5,True)
    def test_active_auk_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'MiMo';root.mkdir()
            state=root.parent/'AuK/results/seedtts-full-seed1234/state.json';state.parent.mkdir(parents=True)
            state.write_text('{"phase":"generating"}')
            with patch.object(common,'ROOT',root):
                with self.assertRaises(RuntimeError):common.assert_auk_idle()

class RunnerTests(unittest.TestCase):
    def test_health_accepts_sglang_empty_body(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
            def log_message(self,*args):pass
        with HTTPServer(('127.0.0.1',0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:http_health(f'http://127.0.0.1:{server.server_port}/health')
            finally:server.shutdown();thread.join()
    def test_transport_errors_retained_and_resume_checked(self):
        cfg=json.loads((common.ROOT/'configs/mmlu-pro.json').read_text())
        items=[{'id':i,'category':'fixture','prompt':f'Fixture {i}','prompt_sha256':str(i),'expected':'A','option_count':4} for i in range(2)]
        calls=[]
        def transport(url,body,timeout):
            calls.append(body)
            if len(calls)==2:raise TimeoutError('synthetic transport failure')
            return raw('The answer is (A).')
        with tempfile.TemporaryDirectory() as td:
            summary=evaluate(cfg,items,td,False,send=transport)
            self.assertEqual(len(calls),2);self.assertEqual(summary['expected'],2)
            self.assertEqual(summary['accuracy_percent'],50);self.assertEqual(summary['errors'],1)
            self.assertNotIn('expected',calls[0])
            self.assertEqual(len((Path(td)/'predictions.jsonl').read_text().splitlines()),2)
            evaluate(cfg,items,td,False,send=transport,resume=True)
            self.assertEqual(len(calls),2)
            with self.assertRaises(AssertionError):evaluate(dict(cfg,max_tokens=1),items,td,False,send=transport,resume=True)

if __name__=='__main__':
    print('CPU fixture tests only; no MiMo inference or benchmark accuracy measurement.',flush=True)
    unittest.main(verbosity=2)
