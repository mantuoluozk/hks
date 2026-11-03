"""Independently execute the original 300 conditions; no reference/language hints."""
import argparse,concurrent.futures as cf,csv,json,os,sys,platform
from pathlib import Path
from asr_api import audio,infer
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir();(a.output/'outputs').mkdir()
items=[json.loads(s) for s in (ROOT/'data/official-items.jsonl').read_text().splitlines()];assert len(items)==300
def run(item):
    r=infer(audio(item['audio_path']),seed=item['seed']);r.update(id=item['id'],condition_sha256=item['condition_sha256'],condition_params=item['params'])
    r['passed']=r['success'] and bool(r['text']) and r.get('finish_reason')!='length'
    if r['success']:(a.output/'outputs'/(item['id']+'.txt')).write_text(r['text']+'\n',encoding='utf-8')
    return r
rows=[]
with cf.ThreadPoolExecutor(4) as pool,(a.output/'predictions.jsonl').open('w',encoding='utf-8') as f:
    for r in pool.map(run,items):rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();print('Official completed',len(rows),'of 300',flush=True)
with (a.output/'反馈表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['condition_id','model','status','output_path','error','parameter_changes'])
    for r in rows:w.writerow([r['id'],'qwen3_asr','成功' if r['passed'] else '失败','outputs/'+r['id']+'.txt',r['error'] or ('truncated/empty' if not r['passed'] else ''),'无'])
report={'expected':300,'completed':len(rows),'successful':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),'all_processed':len(rows)==300,'reference_or_language_supplied':False,'condition_order_preserved':True,'official_acceptance':None}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
(a.output/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'framework':'SGLang','model_path':os.environ['MODEL_PATH'],'runtime_job':os.environ['RUN_RESULTS_DIR'],'temperature':0,'max_new_tokens':1024,'language':None,'dtype':'bf16','seed':42},indent=2)+'\n')
(a.output/'环境说明.txt').write_text('Qwen3-ASR-1.7B × SGLang，单BW；BF16/greedy/1024/自然EOS/自动语种。300条件独立请求，未输入参考文本或语种标签；完整参数和来源见environment.json及逐条记录。\n',encoding='utf-8')
print(json.dumps(report),flush=True);raise SystemExit(0 if report['successful']==300 else 1)
