"""Recompute the C4 experiment and verify warmup, log and owned-stop evidence."""
import hashlib,re,statistics
from pathlib import Path
from evidence_io import read_json as read,read_jsonl
from audit_audio import close
ROOT=Path(__file__).resolve().parents[1]
def check(folder):
    summary=read(folder/'summary.json');protocol=read(folder/'protocol.json')
    assert read(folder/'state.json')['stage']=='completed' and summary['passed']
    assert protocol['warmup_batches']==protocol['measured_batches']==3 and protocol['requests_per_batch']==protocol['concurrency']==4
    all_ids=set();commands=[]
    baseline=read_jsonl(ROOT/'results/eager-template/short/requests.jsonl')
    reference={lang:{r['text'] for r in baseline if r['language_input']==lang} for lang in ['en','zh']}
    for f in read(folder/'source-hashes.json'):
        assert hashlib.sha256((folder/'source'/f['path']).read_bytes()).hexdigest()==f['sha256']
    for mode in ['eager','graph']:
        job=folder/mode;rows=read_jsonl(job/'requests.jsonl');batches=read(job/'batches.json');cell=read(job/'summary.json')
        assert len(rows)==24 and len(batches)==6 and cell==summary['cells'][mode]
        for r in rows:
            assert r['success'] and not r['error'] and r['finish_reason']!='length' and r['text'] in reference[r['language_input']]
            assert r['text']==r['response']['choices'][0]['message']['content'].split('<asr_text>',1)[-1].strip()
            assert r['usage']['completion_tokens']>0 and not r['request']['reference_or_language_supplied']
            close(r['rtf'],r['elapsed_seconds']/r['audio_seconds'])
            assert r['response']['id'] not in all_ids;all_ids.add(r['response']['id'])
        for b in batches:
            matched=[r for r in rows if (r['phase'],r['batch'])==(b['phase'],b['batch'])]
            assert len(matched)==b['requests']==b['concurrency']==4
            assert [r['language_input'] for r in matched]==['en','zh','en','zh']
            close(b['audio_seconds'],sum(r['audio_seconds'] for r in matched));close(b['audio_seconds_per_wall_second'],b['audio_seconds']/b['wall_seconds'])
            assert b['completion_tokens']==sum(r['usage']['completion_tokens'] for r in matched)
            segment=(job/b['server_log_segment']).read_text(encoding='utf-8')
            assert segment in (job/'server.log').read_text(encoding='utf-8')
            warnings=[l for l in segment.splitlines() if re.search(r'took .* to compile|compilation can stall',l,re.I)]
            assert warnings==b['compilation_warnings']
            if b['phase']=='measured' or b['batch'] in [2,3]:assert not warnings
        m=[r for r in rows if r['phase']=='measured'];mb=[b for b in batches if b['phase']=='measured']
        assert len(m)==cell['counted_requests']==12 and cell['warmup_requests']==12 and cell['passed']
        close(cell['audio_seconds'],sum(r['audio_seconds'] for r in m));close(cell['wall_seconds'],sum(b['wall_seconds'] for b in mb));close(cell['audio_seconds_per_wall_second'],cell['audio_seconds']/cell['wall_seconds'])
        close(cell['mean_latency_seconds'],statistics.mean(r['elapsed_seconds'] for r in m));close(cell['mean_request_rtf'],statistics.mean(r['rtf'] for r in m))
        assert cell['batch_throughputs']==[b['audio_seconds_per_wall_second'] for b in mb]
        stop=read(job/'stop-request.json');exit=read(job/'service-exit.json')
        assert stop['phase']=='stopped' and stop['port_free'] and exit['expected_shutdown'] and exit['validated_stop_requested']
        cmd=read(job/'command.json')
        if mode=='eager':cmd.remove('--disable-cuda-graph')
        else:
            i=cmd.index('--cuda-graph-bs-decode');assert cmd[i+1:i+6]==['1','2','4','8','16'];del cmd[i:i+6]
        commands.append(cmd)
    assert commands[0]==commands[1]
    close(summary['graph_over_eager_percent'],100*(summary['cells']['graph']['audio_seconds_per_wall_second']/summary['cells']['eager']['audio_seconds_per_wall_second']-1))
    return {'passed':True,'counted_requests':24,'warmup_requests':24,'modes':2,'last_two_warmups_and_all_measurements_no_compilation_warning':True,'source_command_and_raw_statistics_checked':True,'model_requests_sent':0}
if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    result=check(a.folder)
    (a.folder/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
