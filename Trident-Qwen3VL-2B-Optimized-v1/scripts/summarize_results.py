"""Audit actual SCNet results and record separate functional, accuracy and performance states."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def main():
    scnet=ROOT/'results/scnet'
    release=next(p for p in sorted(scnet.glob('release-*'),reverse=True) if read(p/'status.json')['stage']=='completed')
    smoke=next(p for p in sorted(scnet.glob('validation-*'),reverse=True) if (p/'native_embedding/api-summary.json').exists() and (p/'native_embedding/performance/summary.json').exists() and read(p/'status.json')['stage']=='initial_native_reference_comparison_complete')
    primary=release/'native_reproduction/performance/summary.json'
    reproduction=smoke/'native_embedding/performance/summary.json'
    p,r=read(primary),read(reproduction)
    assert p['cache_policy']==r['cache_policy']=='unique_image_uuid_per_request'
    cells=[]
    for results in [p,r]:
        assert len(results['cells'])==24 and sum(c['requests'] for c in results['cells'])==1536
        assert {(c['round'],c['fixture'],c['batch_size']) for c in results['cells']}=={(round_,kind,batch) for round_ in [1,2] for kind in ['text','image','fused'] for batch in [1,2,4,8]}
    for a,b in zip(p['cells'],r['cells']):
        assert (a['round'],a['fixture'],a['batch_size'],a['requests'],a['input_tokens'])==(b['round'],b['fixture'],b['batch_size'],b['requests'],b['input_tokens'])
        if a['round']!=2:continue
        cells.append({'fixture':a['fixture'],'batch_size':a['batch_size'],'primary_requests_per_second':a['requests_per_second'],
                      'reproduction_requests_per_second':b['requests_per_second'],'deviation_percent':100*(b['requests_per_second']/a['requests_per_second']-1)})
    maximum=max(abs(c['deviation_percent']) for c in cells)
    native=read(smoke/'native_embedding/summary.json');api=read(smoke/'native_embedding/api-summary.json')
    minimum=min(c for row in native['comparisons'] for c in row['cosine_to_publisher_backend'])
    assert minimum>0.99 and all(c for row in native['comparisons'] for c in row['prompt_token_ids_equal'])
    assert api['passed'] and sum(row['count'] for row in api['cases'])==6
    assert read(smoke/'weights/weights-check.json')['all_required_files_match']
    hf=read(release/'publisher_chartqa/origin-result.json');vl=read(release/'native_primary/chartqa/origin-result.json')
    for item in [hf,vl]:
        assert item['num_evaluated_queries']==1250 and item['stats']['corpus_rows']==1509
        assert item['stats']['queries_embedding_failed']==item['stats']['gold_docs_missing_in_corpus']==0
    references=[]
    selected=[release,smoke,ROOT/'results/dataset']
    for directory in selected:
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or f.suffix not in {'.json','.jsonl','.txt','.log'} or f.name=='backup.log':continue
            raw=f.read_bytes();references.append({'path':f.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    result={'model':'Trident-Qwen3VL-2B','framework':'native vLLM','dtype':'bfloat16','official_accuracy_pass':None,
            'functional_pass':True,'functional_inputs':6,'includes_two_images_in_one_input':True,'embedding_api_pass':api['passed'],
            'minimum_cosine_to_publisher_backend':minimum,'all_prompt_token_ids_equal':True,
            'chartqa_origin':{'queries':1250,'corpus':1509,'publisher_on_BW_ndcg_at_10_x100':100*hf['metrics']['ndcg@10'],
                             'native_vllm_ndcg_at_10_x100':100*vl['metrics']['ndcg@10'],'paper_gpu_ndcg_at_10_x100':70.19,
                             'native_minus_publisher_BW_pt':100*(vl['metrics']['ndcg@10']-hf['metrics']['ndcg@10']),
                             'native_minus_paper_pt':100*vl['metrics']['ndcg@10']-70.19},
            'performance':{'primary':primary.relative_to(ROOT).as_posix(),'reproduction':reproduction.relative_to(ROOT).as_posix(),
                           'requests_per_process':1536,'cells_per_process':24,'report_round':2,'comparisons':cells,
                           'maximum_absolute_deviation_percent':maximum,'within_20_percent':maximum<=20,
                           'scope':'Same Notebook and image, independently initialized model processes; no fresh-instance claim.'},
            'release_path':release.relative_to(ROOT).as_posix(),'smoke_path':smoke.relative_to(ROOT).as_posix(),
            'preliminary_cached_probe':(release/'native_primary/performance').relative_to(ROOT).as_posix(),
            'image':{'url':'https://www.scnet.cn/ui/aihub/image/zack233/2114624281480511490/versions',
                     'registry_reference':'appstore.scnet.cn:5000/aihubprivate/dcu/zack233/qwen-asr:v1','version':'v1',
                     'digest':None,'metadata_deferred_by_owner':True,'region':'华中一区（A区）','visibility':'private'},
            'evidence_audit_passed':True,'references':references}
    (scnet/'validation-summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k!='references'},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
