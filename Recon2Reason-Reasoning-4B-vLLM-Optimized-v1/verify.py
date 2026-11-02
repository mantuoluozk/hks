"""Recompute optimization evidence and validate the released file inventory."""
import argparse,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def smoke(folder):
 summary=read(folder/'summary.json');assert len(summary['cases'])==5
 answers={}
 for case in summary['cases']:
  label=case['case'];raw=(folder/f'{label}.response.txt').read_text(encoding='utf-8')
  if label=='stream_image':
   events=[x[6:] for x in raw.splitlines() if x.startswith('data: ')]
   assert '[DONE]' in events
   data=[json.loads(x) for x in events if x!='[DONE]']
   answer=''.join(x['choices'][0]['delta'].get('content') or '' for x in data if x.get('choices'))
  else:answer=json.loads(raw)['choices'][0]['message']['content']
  assert answer==case['answer'] and answer.strip()
  answers[label]=answer.strip().lower()
 assert re.fullmatch(r'5[.!]?',answers['text'])
 assert re.search(r'\bred\b',answers['single_image'])
 assert re.findall(r'\b(?:red|blue)\b',answers['multi_image'])[:2]==['red','blue']
 assert re.search(r'\bblue\b',answers['multi_turn']) and re.search(r'\bred\b',answers['stream_image'])
 return {'path':folder.name,'checks':5,'passed':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--skip-manifest',action='store_true');a=p.parse_args()
 name=Path(a.job or read(ROOT/'results/formal-current.json')['path']).name
 for script,params in [('scripts/verify_screen.py',['--job','optimization-20261102T042055Z']),('verify_optimized.py',['--job',name]),('scripts/audit_backend.py',['--job',name])]:
  subprocess.run([sys.executable,str(ROOT/script),*params],check=True,cwd=ROOT)
 if not a.skip_manifest:subprocess.run([sys.executable,str(ROOT/'scripts/check_release.py')],check=True,cwd=ROOT)
 print('Optimization evidence verified; official acceptance is not asserted.')
if __name__=='__main__':main()
