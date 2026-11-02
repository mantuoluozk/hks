"""Verify released evidence without treating self-tests as official accuracy approval."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from PIL import Image, ImageStat

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--selfcheck',action='store_true')
a=p.parse_args()


def read(relative):
    path=(ROOT/relative).resolve()
    assert path.is_relative_to(ROOT.resolve()),relative
    return json.loads(path.read_text(encoding='utf-8'))


try:
    manifest=read('results/evidence-manifest.json')
    for name,digest in manifest['sha256'].items():
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT.resolve()),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    weights=read(manifest['weights'])
    assert weights['all_runtime_files_match'] and len(weights['files'])==17
    assert read(manifest['configuration'])['configuration_equivalent']
    cases=read(manifest['functional'])['cases']
    assert len(cases)==8 and all(r['http_status']==200 for r in cases)
    by_name={r['case']:r for r in cases}
    assert by_name['text512-repeat']['same_seed_pixels_identical']
    assert by_name['text512-otherseed']['different_seed_pixels_differ']
    assert by_name['text512']['pixel_sha256']==by_name['text512-repeat']['pixel_sha256']
    assert by_name['text512']['pixel_sha256']!=by_name['text512-otherseed']['pixel_sha256']
    assert by_name['edit-one']['condition_image_count']==1
    assert by_name['edit-two']['condition_image_count']==2
    image_dir=(ROOT/manifest['functional']).resolve().parent
    for case in cases:
        width,height=map(int,case['size'].split('x'))
        path=(image_dir/(case['case']+'.png')).resolve()
        assert path.is_relative_to(image_dir),case['case']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['image_sha256'],case['case']
        with Image.open(path) as image:
            image.load()
            assert image.size==(width,height) and image.mode==case['image_mode'],case['case']
            assert hashlib.sha256(image.tobytes()).hexdigest()==case['pixel_sha256'],case['case']
            assert max(ImageStat.Stat(image.convert('RGB')).stddev)>1.0,case['case']
    # Bind the API inputs to the backend kwargs actually received in the worker.
    trace=[]
    for name in manifest['request_traces']:
        trace.extend(json.loads(l) for l in (ROOT/name).read_text().splitlines())
    for case in cases:
        width,height=map(int,case['size'].split('x'))
        assert any(r['prompt']==case['prompt'] and r['seed']==case['seed']
            and r['width']==width and r['height']==height
            and r['generator_devices']==['cpu'] and r['true_cfg_scale']==1.0
            and r['condition_image_count']==case['condition_image_count'] for r in trace),case['case']
    validation=read(manifest['service_validation'])
    assert validation['status']=='completed'
    paired=validation['performance_comparison'];assert len(paired)==3

    def round_means(directory):
        rows=[json.loads(l) for l in (directory/'benchmark/requests.jsonl').read_text().splitlines() if l.strip()]
        assert len(rows)==12,directory
        means={}
        for size in (512,1024,2048):
            group=[r for r in rows if r['size']==size]
            assert sorted(r['iteration'] for r in group)==[0,1,2,3],size
            warm=[r for r in group if r['warmup'] is True]
            measured=[r for r in group if r['warmup'] is False]
            assert len(warm)==1 and warm[0]['iteration']==0 and len(measured)==3,size
            seconds=[float(r['request_seconds']) for r in group]
            assert all(math.isfinite(s) and s>0 for s in seconds),size
            means[size]=sum(float(r['request_seconds']) for r in measured)/3
        return means

    base=(ROOT/manifest['service_validation']).resolve().parent
    first,second=round_means(base/'round-1'),round_means(base/'round-2')
    computed={s:abs(second[s]-first[s])/first[s]*100 for s in first}
    seen={}
    for entry in paired:
        size=entry['size']
        assert math.isclose(entry['first_mean_seconds'],first[size]),entry
        assert math.isclose(entry['second_mean_seconds'],second[size]),entry
        assert math.isclose(entry['relative_difference_percent'],computed[size]),(size,entry['relative_difference_percent'],computed[size])
        seen[size]=entry
    assert set(seen)=={512,1024,2048}
    reproduction=all(v<=20 for v in computed.values())
    image_verified=False
    if manifest.get('image_info'):
        sys.path.insert(0,str(ROOT/'scripts'))
        from image_gate import evaluate_image
        image=evaluate_image(ROOT,base,{k:manifest[k] for k in ['image_preflight','image_install_check','image_info']})
        image_verified=image['dedicated_image_revalidation']
        assert image_verified==manifest['dedicated_image_revalidation']
    selfcheck=reproduction and (image_verified if manifest.get('image_info') else True)
    result={'evidence_hashes_pass':True,'functional_pass':True,
            'performance_reproduction_within_20_percent':reproduction,
            'selfcheck_pass':selfcheck,'official_accuracy_pass':None,
            'dedicated_image_revalidation':image_verified,
            'official_acceptance_pass':None}
    code=0 if selfcheck else 2
except Exception as exc:
    result={'selfcheck_pass':False,'error':str(exc),'official_accuracy_pass':None,'official_acceptance_pass':None}
    code=1
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/verify_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));sys.exit(code)
