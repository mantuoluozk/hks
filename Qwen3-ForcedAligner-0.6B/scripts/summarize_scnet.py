"""Audit the actual ForcedAligner first benchmark and fresh-process reproduction."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())


def audit(folder):
    assert read(folder/'status.json')['stage']=='functional_and_performance_complete'
    weights=read(folder/'runtime/weights-check.json')
    assert weights['all_required_files_match'] and len(weights['files'])==9
    preflight=read(folder/'preflight/summary.json')
    assert preflight['native_token_classification_import'] and preflight['weights_available']
    smoke=read(folder/'smoke/summary.json')
    assert smoke['functional_cases']==2 and smoke['max_num_seqs']==8 and smoke['enforce_eager']
    assert {x['language'] for x in smoke['cases']}=={'Chinese','English'}
    assert all(x['format_monotonicity_bounds_passed'] for x in smoke['cases'])
    bench=read(folder/'smoke/bench/summary.json')
    rows=[json.loads(x) for x in (folder/'smoke/bench/requests.jsonl').read_text().splitlines()]
    assert len(bench['workloads'])==8 and bench['counted_requests']==len(rows)==512
    for w in bench['workloads']:
        subset=[x for x in rows if x['round']==w['round'] and x['batch_size']==w['batch_size']]
        assert len(subset)==w['requests'] and all(x['format_monotonicity_bounds_passed'] for x in subset)
        audio=sum(x['audio_seconds'] for x in subset)
        assert abs(audio-w['audio_seconds'])<1e-8
        assert abs(audio/w['wall_seconds']-w['audio_seconds_per_wall_second'])<1e-8
    return {'preflight':preflight,'smoke':smoke,'performance':bench}


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',required=True);p.add_argument('--reproduction',required=True);args=p.parse_args()
    first=audit(ROOT/args.primary);second=audit(ROOT/args.reproduction)
    assert first['preflight']['packages']==second['preflight']['packages']
    assert first['preflight']['devices']==second['preflight']['devices']
    assert first['preflight']['native_aligner_source_sha256']==second['preflight']['native_aligner_source_sha256']
    assert first['preflight']['cpu_thread_environment']==second['preflight']['cpu_thread_environment']
    cross_process=[]
    for a in first['smoke']['cases']:
        b=next(x for x in second['smoke']['cases'] if x['id']==a['id'])
        assert [x['text'] for x in a['items']]==[x['text'] for x in b['items']]
        cross_process.append({'id':a['id'],'exact_match':a['items']==b['items'],
                              'maximum_delta_seconds':max(abs(x[k]-y[k]) for x,y in zip(a['items'],b['items']) for k in ['start_time','end_time'])})
    comparison=[]
    for a in first['performance']['workloads']:
        if a['round']!=2:continue
        b=next(x for x in second['performance']['workloads'] if x['round']==2 and x['batch_size']==a['batch_size'])
        delta=100*(b['audio_seconds_per_wall_second']/a['audio_seconds_per_wall_second']-1)
        comparison.append({'batch_size':a['batch_size'],'primary_audio_seconds_per_wall_second':a['audio_seconds_per_wall_second'],
                           'reproduction_audio_seconds_per_wall_second':b['audio_seconds_per_wall_second'],'deviation_percent':delta})
    refs=[]
    for name in [args.primary,args.reproduction]:
        for path in sorted((ROOT/name).rglob('*')):
            if not path.is_file() or 'backup' in path.name or path.suffix not in {'.json','.jsonl','.log'}:continue
            raw=path.read_bytes();refs.append({'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    summary={'model':'Qwen/Qwen3-ForcedAligner-0.6B','evidence_audit_passed':True,
             'primary_path':args.primary,'reproduction_path':args.reproduction,'primary':first,'reproduction':second,
             'performance_reproduction':comparison,'maximum_absolute_deviation_percent':max(abs(x['deviation_percent']) for x in comparison),
             'performance_reproduction_within_twenty_percent':all(abs(x['deviation_percent'])<=20 for x in comparison),
             'image':read(ROOT/'sources/scnet-image.json'),'references':refs,
             'cross_process_functional_comparison':cross_process,
             'official_accuracy_pass':None,'official_acceptance_passed':None,
             'reproduction_scope':'Two independent processes in the same SCNet Notebook and image; not a new Notebook instance.'}
    (ROOT/'results/scnet/validation-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in {'primary','reproduction','references'}},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
