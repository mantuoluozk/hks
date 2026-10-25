"""Run public self-evaluation, then assemble auditable verify_result.json.

Official GPU baseline and SCNet acceptance stay pending; no fabricated PASS.
"""
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(directory):
    on_scnet=os.environ.get('EXECUTION_PLATFORM')=='SCNet Notebook'
    result={'model':'BAAI/Recon2Reason-Reasoning-4B','hardware':'SCNet BW 64GB / 80 CU' if on_scnet else 'own BW100',
            'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned BW100'),
            'purpose':'public zero-shot self-evaluation with original released processor',
            'official_scnet_validation':False,'official_gpu_baseline_received':False,
            'baseline_tolerance_pass':None,'datasets':{},'limits':[],
            'publisher_baseline_lookup':{'status':'no_numeric_results_found_in_checked_official_sources',
                'report':'官方精度基线查证.md',
                'retrieval_record':'sources/baseline-search-20261023/retrieval.json'}}
    for dataset,expected in [('ocrbench',1000),('mmmu',900)]:
        folder=directory/dataset
        summary=read(folder/'summary.json');manifest=read(folder/'manifest.json')
        scored=[json.loads(s) for s in (folder/'scored.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        raw=[json.loads(s) for s in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        assert len(raw)==len(scored)==summary['count']==expected
        assert {r['id'] for r in scored}==set(manifest['ids'])=={r['id'] for r in raw}
        assert len(set(manifest['ids']))==expected
        assert summary['full_dataset']
        assert sum(bool(r['correct']) for r in scored)==summary['correct']
        assert sum(bool(r['strict_correct']) for r in scored)==summary['strict_correct']
        assert sum(bool(r['error']) for r in scored)==summary['errors']
        assert sum(r.get('finish_reason')=='length' for r in scored)==summary['truncated']
        result['datasets'][dataset]=dict(summary,artifact_directory=str(folder.relative_to(ROOT)),
            dataset_revision=manifest['dataset']['revision'],generation=manifest['generation'])
        if summary['truncated']:
            result['limits'].append(f"{dataset}: {summary['truncated']} outputs reached max_tokens={manifest['generation']['max_tokens']}; included in score")
    result['local_evaluation_complete']=all(x['complete'] and x['errors']==0 for x in result['datasets'].values())
    result['limits'].extend(['No official GPU baseline: tolerance cannot be decided.',
        'Official acceptance and final model image publication pending.' if on_scnet else 'SCNet formal rerun and repository/image publication pending.',
        'Original max_pixels=50176; no client-side image resizing or model changes.'])
    output=ROOT/'results/scnet/verify_result.json' if on_scnet else ROOT/'verify_result.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if on_scnet:
        primary=ROOT/'verify_result.json'
        historical=ROOT/'repro/self-owned-prevalidation/verify_result.json'
        if primary.exists() and not historical.exists() and read(primary).get('execution_platform')!='SCNet Notebook':
            historical.parent.mkdir(parents=True,exist_ok=True)
            historical.write_bytes(primary.read_bytes())
        primary.write_bytes(output.read_bytes())
    print(json.dumps({k:result[k] for k in ['local_evaluation_complete','baseline_tolerance_pass','limits']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/public-accuracy-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    directory=(ROOT/args.output).resolve()
    assert directory.is_relative_to(ROOT)
    if not args.summarize_only:
        subprocess.run(['bash','verify.sh'],cwd=ROOT,env=dict(os.environ,EVAL_OUT=str(directory)),check=True)
    summarize(directory)
