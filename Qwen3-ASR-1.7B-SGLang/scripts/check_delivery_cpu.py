"""Exercise saved-evidence failure exits in an isolated copy; no model calls."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
work=Path(tempfile.mkdtemp(prefix='asr-sglang-delivery-'))
copy=work/'model'
shutil.copytree(ROOT,copy,ignore=shutil.ignore_patterns('cache','artifacts','weights','data','__pycache__','.git','*.pyc'))
(copy/'data').mkdir();shutil.copyfile(ROOT/'data/README.md',copy/'data/README.md')
cases=[]
def run(name,expected):
    p=subprocess.run([sys.executable,'verify.py'],cwd=copy,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (work/(name+'.log')).write_bytes(p.stdout)
    cases.append({'case':name,'actual_exit':p.returncode,'expected_exit':expected,'passed':p.returncode==expected})
    assert p.returncode==expected,(name,p.returncode,p.stdout.decode(errors='replace'))
run('valid',0)
p=copy/'results/release.json';original=p.read_bytes();p.unlink()
try:run('missing_release',1)
finally:p.write_bytes(original)
p=next((copy/'assets/evidence').glob('*.zlib'));original=p.read_bytes()
try:
    p.write_bytes(bytes([original[0]^1])+original[1:]);run('corrupted_asset',1)
finally:p.write_bytes(original)
p=copy/'env.log';original=p.read_bytes()
try:
    p.write_bytes(original+b'\nCPU replay output fixture\n');run('mutable_env_log',0)
finally:p.write_bytes(original)
result={'passed':True,'model_requests_sent':0,'cases':cases,'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'isolated_logs_retained':True}
(ROOT/'results/delivery-cpu-checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result));print('PRIVATE_CHECK_LOGS',work)
