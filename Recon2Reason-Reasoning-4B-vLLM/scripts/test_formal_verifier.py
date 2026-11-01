"""The special-token exception must reject altered or unsupported evidence."""
import copy,importlib.util,json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import verify_formal as verifier
with tempfile.TemporaryDirectory() as tmp:
 work=Path(tmp)
 for name in ('sources/publisher-tokenizer-config.json','results/empty-output-diagnosis/publisher-reference.json',
  *[f'results/empty-output-diagnosis/full-index{i}-skip{s}.response.json' for i in (71,98) for s in (True,False)]):
  dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
 verifier.ROOT=work
 assert set(verifier.confirmed_special_only())=={71,98}
 target=work/'results/empty-output-diagnosis/full-index71-skipTrue.response.json'
 original=json.loads(target.read_text());bad=copy.deepcopy(original)
 bad['choices'][0]['token_ids'][0]=0
 target.write_text(json.dumps(bad))
 try:verifier.confirmed_special_only()
 except AssertionError:pass
 else:raise AssertionError('Altered backend tokens accepted')
 target.write_text(json.dumps(original))
 ref=work/'results/empty-output-diagnosis/publisher-reference.json'
 bad=json.loads(ref.read_text());bad[0]['new_tokens'][0]=0;ref.write_text(json.dumps(bad))
 try:verifier.confirmed_special_only()
 except AssertionError:pass
 else:raise AssertionError('Altered publisher reference accepted')
print('Special-token proof positive and two negative controls passed')
