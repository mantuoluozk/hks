"""CPU-only contract tests: mock HTTP service, temporary package and shell fixtures."""
import importlib.util,json,os,shutil,subprocess,sys,tempfile,threading,unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import bench_vllm,prepare_runtime

class Handler(BaseHTTPRequestHandler):
    mode='normal'
    def log_message(self,*args):pass
    def do_POST(self):
        q=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.mode=='error':
            self.send_response(500);self.end_headers();self.wfile.write(b'failure');return
        if q['model']!='test-model' or q['max_tokens']<0:
            self.send_response(400);self.end_headers();self.wfile.write(b'invalid');return
        prompt=q['messages'][-1]['content']
        answer='READY' if 'READY' in prompt else '6' if 'Add one' in prompt else '5'
        if self.mode=='degenerate':answer='!!!!'
        self.send_response(200);self.end_headers()
        if q.get('stream'):
            events=[{'choices':[{'delta':{'content':answer}}]}]
            if self.mode!='no_usage':events.append({'choices':[],'usage':{'completion_tokens':1}})
            for e in events:self.wfile.write(('data: '+json.dumps(e)+'\n\n').encode())
            if self.mode!='no_done':self.wfile.write(b'data: [DONE]\n\n')
        else:self.wfile.write(json.dumps({'choices':[{'message':{'content':answer},'finish_reason':'stop'}],'usage':{'completion_tokens':1}}).encode())

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='kimi-entry-')
        self.work=Path(self.temp.name).resolve()
        assert self.work.is_relative_to(Path(tempfile.gettempdir()).resolve())
        Handler.mode='normal'
    def tearDown(self):self.temp.cleanup()
    def cli(self,script,*args):
        return subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],env={**os.environ,'PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8',timeout=30)
    def test_smoke_positive_custom_endpoint_and_evidence(self):
        out=self.work/'smoke';r=self.cli('vllm_smoke.py','--output',out,'--base',self.base,'--model','test-model')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        s=json.loads((out/'summary.json').read_text());self.assertTrue(s['passed']);self.assertEqual(len(s['checks']),6)
        self.assertTrue((out/'stream.response.txt').exists())
        before=(out/'summary.json').read_bytes()
        self.assertNotEqual(self.cli('vllm_smoke.py','--output',out,'--base',self.base).returncode,0)
        self.assertEqual((out/'summary.json').read_bytes(),before)
    def test_smoke_rejects_server_error_degenerate_and_truncated_stream(self):
        for mode in ('error','degenerate','no_done'):
            Handler.mode=mode;out=self.work/mode
            r=self.cli('vllm_smoke.py','--output',out,'--base',self.base,'--model','test-model')
            self.assertNotEqual(r.returncode,0,mode)
            self.assertFalse(json.loads((out/'summary.json').read_text())['passed'])
    def test_benchmark_arguments_and_backend_usage(self):
        out=self.work/'bench'
        r=self.cli('bench_vllm.py','--output',out,'--base',self.base,'--model','test-model')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        summary=json.loads((out/'summary-run.json').read_text())
        self.assertEqual(len(summary['points']),6)
        self.assertTrue(all(x['output_tokens_total']==x['requests'] for x in summary['points']))
        self.assertTrue(all(x['mean_tpot_ms'] is None for x in summary['points']))
        records=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),60)
        Handler.mode='no_usage'
        with self.assertRaises(ValueError):bench_vllm.one('test',128,stream=True,base=self.base,model='test-model')
        Handler.mode='no_done'
        with self.assertRaises(ValueError):bench_vllm.one('test',128,stream=True,base=self.base,model='test-model')
    def fake_package(self,version,symbol=True):
        dest=self.work/f'pkg-{version}-{symbol}';p=dest/'openai/types/responses';p.mkdir(parents=True)
        (dest/'openai/__init__.py').write_text(f"__version__={version!r}\n")
        (dest/'openai/types/__init__.py').write_text('')
        (p/'__init__.py').write_text('class NamespaceTool: pass\n' if symbol else '')
        return dest
    def test_runtime_rejects_old_version_and_missing_symbol(self):
        self.assertFalse(prepare_runtime.check(self.fake_package('2.6.1')))
        self.assertFalse(prepare_runtime.check(self.fake_package('2.30.0',False)))
        self.assertTrue(prepare_runtime.check(self.fake_package('2.30.0')))
    def test_patch_fixture_idempotent_and_exact_source(self):
        script=ROOT/'runtime/apply_vllm_kda_fix.py'
        spec=importlib.util.spec_from_file_location('kimi_patch_fixture',script)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for rel,changes in mod.CHANGES.items():
            p=self.work/rel;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('\n'.join(before+'\n' for before,after,count in changes for _ in range(count)))
        def run(*args):return subprocess.run([sys.executable,str(script),'--site-packages',str(self.work),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
        self.assertNotEqual(run('--check').returncode,0)
        self.assertEqual(run().returncode,0)
        self.assertEqual(run().returncode,0)
        self.assertEqual(run('--check').returncode,0)
        for rel in mod.CHANGES:self.assertTrue((self.work/(rel+'.kimi-linear-original')).exists())
        first=self.work/next(iter(mod.CHANGES))
        first.write_text('unsupported upstream source\n')
        self.assertNotEqual(run().returncode,0)
        self.assertEqual(first.read_text(),'unsupported upstream source\n')
    def test_shell_launcher_fails_closed_before_api(self):
        bash=os.environ.get('TEST_BASH') or shutil.which('bash')
        if not bash:self.skipTest('Bash not available')
        for n in ('run.sh','env.sh'):shutil.copy2(ROOT/n,self.work/n)
        wrapper='''#!/usr/bin/env bash
printf '%s\\n' "$*" >> calls.log
case "$*" in
 *runtime/apply_vllm_kda_fix.py*) exit "${MOCK_PATCH_EXIT:-0}" ;;
 *scripts/check_weights.py*) exit "${MOCK_WEIGHTS_EXIT:-0}" ;;
 *) echo API_LAUNCHED >> calls.log; exit 0 ;;
esac
'''
        (self.work/'run-scnet.sh').write_text(wrapper)
        def run(**extra):
            e={**os.environ,'MOCK_WEIGHTS_EXIT':'2',**extra}
            return subprocess.run([bash,'run.sh'],cwd=self.work,env=e,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        self.assertEqual(run().returncode,2)
        self.assertNotIn('API_LAUNCHED',(self.work/'calls.log').read_text())
        self.assertEqual(run(MOCK_PATCH_EXIT='4').returncode,3)
        self.assertEqual(run(MOCK_WEIGHTS_EXIT='0').returncode,0)
        calls=(self.work/'calls.log').read_text()
        self.assertIn('vllm.entrypoints.openai.api_server',calls);self.assertNotIn('sglang.launch_server',calls)
        for f in ROOT.glob('*.sh'):
            result=subprocess.run([bash,'-n',str(f)],capture_output=True,text=True,encoding='utf-8',errors='replace')
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
