"""Saved-evidence integrity/statistics checks only; no online inference or scoring."""
import csv,hashlib,io,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'scripts'))
from evidence_io import read_json as read,read_jsonl,read_text
from audit_audio import check,close
def functional(folder):
    s=read(folder/'functional/summary.json');assert s['passed'] and s['total']==s['successful']==12
    assert all(s['repeat_exact'].values()) and all(s['sse_chat_exact'].values())
    for p in (folder/'functional').glob('*.json'):
        if p.name=='summary.json':continue
        r=read(p);assert r['passed'] and r['success'] and r['text'] and not r['error']
        if r['mode']=='sse':assert r['response']['done'] and r['response']['raw_sse_lines'][-1]=='data: [DONE]'
def accuracy(folder):
    rows=read_jsonl(folder/'predictions.jsonl');s=read(folder/'summary.json');proof=read(ROOT/'results/test-clean-inputs.json')
    assert proof['passed'] and proof['count']==2620 and proof['source_sha256']=='7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0'
    inputs={r['id']:r for r in proof['files']};assert len(rows)==len(inputs)==2620 and {r['id'] for r in rows}==set(inputs)
    assert s['complete'] and s['requested']==s['completed']==s['successful']==2620 and s['request_errors']==s['truncated']==0
    for r in rows:
        assert r['success'] and not r['error'] and r['finish_reason']!='length' and r['sha256']==inputs[r['id']]['sha256']
        assert r['text']==r['response']['choices'][0]['message']['content'].split('<asr_text>',1)[-1].strip()
        assert r['word_errors']==r['substitutions']+r['deletions']+r['insertions']
        assert r['request']['reference_or_language_supplied'] is False and r['usage']['completion_tokens']>0
    for key in ['reference_words','word_errors','substitutions','deletions','insertions','reference_chars','char_errors']:assert s[key]==sum(r[key] for r in rows),key
    close(s['wer_percent'],100*s['word_errors']/s['reference_words']);close(s['cer_percent'],100*s['char_errors']/s['reference_chars'])
    re=read(folder/'rescore.json');assert re['passed'] and re['rows_checked']==2620 and re['model_requests_sent']==0 and re['reference_and_language_not_sent']
    assert s['official_acceptance'] is None and not s['publisher_normalization_equivalence_verified']
    ids={r['response']['id'] for r in rows};assert len(ids)==2620
    return s,ids
def official(folder):
    rows=read_jsonl(folder/'predictions.jsonl');p=read(ROOT/'results/official-inputs.json');inputs=p['files'];s=read(folder/'summary.json')
    assert p['passed'] and p['count']==300 and len(rows)==300 and [r['id'] for r in rows]==[r['id'] for r in inputs]
    assert s['successful']==s['expected']==s['completed']==300 and s['failed']==0 and s['all_processed']
    for r,i in zip(rows,inputs):
        assert r['success'] and r['passed'] and r['text'] and not r['error'] and r['finish_reason']!='length'
        assert r['sha256']==i['sha256'] and r['condition_sha256']==i['condition_sha256'] and r['condition_params']==i['params']
        assert r['request']['reference_or_language_supplied'] is False
        assert (folder/'outputs'/(r['id']+'.txt')).read_text(encoding='utf-8')==r['text']+'\n'
    feedback=list(csv.DictReader(io.StringIO((folder/'反馈表.csv').read_text(encoding='utf-8-sig'))))
    assert [r['condition_id'] for r in feedback]==[r['id'] for r in inputs] and all(r['status']=='成功' and r['parameter_changes']=='无' and not r['error'] for r in feedback)
    re=read(folder/'rescore.json');assert re['passed'] and re['rows_checked']==300 and re['reference_and_language_not_sent']
    ids={r['response']['id'] for r in rows};assert len(ids)==300
    return s,ids
def main():
    release=read(ROOT/'results/release.json');job=ROOT/'results'/release['formal'];selection=read(ROOT/'results/selection.json')
    assert read(job/'state.json')['stage']=='completed' and (ROOT/'results/formal.exit').read_text().strip()=='0'
    weights=read(ROOT/'results'/release['preflight']/'weights-check.json');assert weights['passed'] and weights['count']==11
    expected={r['Path']:r for r in read(ROOT/'sources/weight-files.json')['Data']['Files'] if r['Type']=='blob' and r['Path'] not in ['README.md','.gitattributes']}
    assert len(weights['files'])==len(expected)==11
    for r in weights['files']:assert (r['bytes'],r['sha256'])==(expected[r['path']]['Size'],expected[r['path']]['Sha256'])
    assert read(ROOT/'results/entry-preparation/environment.json')['environment_source'] in ['image-platform','image-dtk']
    assert (ROOT/'results/entry-preparation.exit').read_text().strip()=='0'
    assert selection['passed'] and selection['same_short_text_sets']
    for mode in ['eager','graph']:
        folder=ROOT/selection['candidate_paths'][mode];functional(folder);check(folder/'short',short=True)
        assert selection['cells'][mode]==read(folder/'short/summary.json')['cells']
    texts=[]
    for mode in ['eager','graph']:
        rows=read_jsonl(ROOT/selection['candidate_paths'][mode]/'short/requests.jsonl')
        texts.append({lang:sorted({r['text'] for r in rows if r['language_input']==lang}) for lang in ['en','zh']})
    assert texts[0]==texts[1]
    functional(job);check(job/'bench')
    independent=read(job/'bench/independent-check.json');assert independent['passed'] and independent['warm_wire_bodies_checked']==16
    a,aid=accuracy(job/'accuracy');o,oid=official(job/'official');assert not aid.intersection(oid)
    template=read(job/'template-preparation.json');assert template['passed'] and template['empty_system_and_audio_render_verified']
    stop=read(job/'stop-request.json');exit=read(job/'service-exit.json');assert stop['phase']=='stopped' and stop['port_free'] and exit['expected_shutdown'] and exit['validated_stop_requested']
    assert 'OWNED_SERVICE_STOPPED' in (job/'stop.log').read_text()
    for name in ['round1.jsonl','round2.jsonl','summary.json']:assert (job/'bench'/name).read_bytes()==(ROOT/'bench/saved-release'/name).read_bytes()
    for stage in ['bench','accuracy','official']:
        current=read(ROOT/'results'/(stage+'-current.json'));latest=ROOT/'results'/Path(current['path']).name
        assert read(latest/'state.json')['stage']=='completed'
        if stage=='bench':
            check(latest/'bench')
            for name in ['round1.jsonl','round2.jsonl','summary.json']:assert (latest/'bench'/name).read_bytes()==(ROOT/'bench'/name).read_bytes()
        elif stage=='accuracy':accuracy(latest/stage)
        else:official(latest/stage)
    for name in ['env.log','server.log','smoke.log']:assert (ROOT/name).is_file()
    for item in read(ROOT/'submission-manifest.json')['files']:
        p=ROOT/item['path'];assert p.resolve().is_relative_to(ROOT)
        assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
    if (ROOT/'sources/evidence-view-map.json').exists():assert read(ROOT/'sources/evidence-view-validation.json')['passed']
    cpu=read(ROOT/'results/delivery-cpu-checks.json');assert cpu['passed'] and cpu['model_requests_sent']==0
    assert [(r['case'],r['actual_exit'],r['expected_exit'],r['passed']) for r in cpu['cases']]==[('valid',0,0,True),('missing_release',1,1,True),('corrupted_asset',1,1,True),('mutable_env_log',0,0,True)]
    assert cpu['checker_sha256']==hashlib.sha256((ROOT/'scripts/check_delivery_cpu.py').read_bytes()).hexdigest()
    result={'saved_evidence_checks_passed':True,'functional_per_validated_service':12,'valid_functional_services':3,'counted_audio_performance_requests':128,'performance_warmup_requests':16,
            'test_clean_utterances':2620,'wer_percent':a['wer_percent'],'cer_percent':a['cer_percent'],'official_conditions_completed':300,
            'online_tests_executed_in_this_command':False,'native_service_actual_exit':exit['actual_returncode'],'validated_stop_requested':True,
            'independent_service_full_reproduction':None,'official_acceptance':None}
    (ROOT/'verify_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'verify_result.json').write_text(json.dumps({'saved_evidence_checks_passed':False,'error':repr(e),'official_acceptance':None},indent=2)+'\n',encoding='utf-8',newline='\n');raise
