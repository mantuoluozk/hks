"""Guard scoring/failed-request semantics and multi-image prompt construction."""
import io
import unittest
from PIL import Image
import verify


class EvaluationProtocolTests(unittest.TestCase):
    def test_ocr_case_and_formula_rules(self):
        row={'answer':['CENTRE'],'dataset':'IIIT5K'}
        record={'output':'centre','error':None}
        self.assertTrue(verify.grade(row,record,'ocrbench')['correct'])
        formula={'answer':['x + Y'],'dataset':'HME100k'}
        self.assertTrue(verify.grade(formula,dict(record,output='x+Y'),'ocrbench')['correct'])
        self.assertFalse(verify.grade(formula,dict(record,output='x+y'),'ocrbench')['correct'])

    def test_failed_request_never_gets_credit(self):
        row={'answer':['CENTRE'],'dataset':'IIIT5K'}
        self.assertFalse(verify.grade(row,{'output':'CENTRE','error':'timeout'},'ocrbench')['correct'])

    def test_mmmu_official_parser_and_strict_fallback(self):
        row={'_id':'test_mc','question_type':'multiple-choice','answer':'B','options':"['alpha','beta']"}
        result=verify.grade(row,{'output':'(B)','error':None},'mmmu')
        self.assertTrue(result['correct']); self.assertTrue(result['strict_correct'])
        fallback=verify.grade(row,{'output':'Unknown','error':None},'mmmu')
        self.assertTrue(fallback['random_fallback']); self.assertFalse(fallback['strict_correct'])
        self.assertEqual(fallback['parsed'],verify.grade(row,{'output':'Unknown','error':None},'mmmu')['parsed'])

    def test_mmmu_open_numeric(self):
        row={'question_type':'open','answer':'1,200'}
        self.assertTrue(verify.grade(row,{'output':'The answer is 1200.','error':None},'mmmu')['correct'])

    def test_interleaved_options_images_no_gold_leak(self):
        stream=io.BytesIO(); Image.new('RGB',(16,16),'red').save(stream,format='PNG')
        image={'bytes':stream.getvalue(),'path':None}
        row={'_id':'test_images','question':'Compare <image 2> to <image 1>.',
             'question_type':'multiple-choice','options':"['yes','no']",'image_1':image,'image_2':image,
             'answer':'SECRET_GOLD','explanation':'SECRET_EXPLANATION'}
        content,trace=verify.make_content(row,'mmmu')
        self.assertEqual([x['key'] for x in trace['images']],['image_2','image_1'])
        self.assertEqual(sum(x['type']=='image_url' for x in content),2)
        self.assertNotIn('SECRET',str(content))


if __name__=='__main__':
    unittest.main()
