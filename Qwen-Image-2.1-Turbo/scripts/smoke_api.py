"""Exercise native SGLang image APIs, preserving actual responses and PNG evidence."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import requests

ROOT=Path(__file__).resolve().parents[1]
out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
base=os.environ.get('BASE_URL','http://127.0.0.1:30003')
session=requests.Session();session.trust_env=False
for _ in range(180):
    try:
        if session.get(base+'/health',timeout=3).status_code==200:break
    except requests.RequestException:pass
    time.sleep(5)
else:raise TimeoutError('SGLang readiness timeout')
models=session.get(base+'/v1/models',timeout=10).json();model=models['data'][0]['id']
(out/'models-response.json').write_text(json.dumps(models,indent=2))
rows=[];arrays={}


def call(name,prompt,size,images=None,seed=42):
    payload={'model':model,'prompt':prompt,'size':size,'n':1,'response_format':'b64_json','seed':seed,
             'generator_device':'cpu','num_inference_steps':8,'true_cfg_scale':1.0,'output_format':'png'}
    start=time.perf_counter()
    if images:
        files=[('image' if len(images)==1 else 'image[]',(p.name,p.read_bytes(),'image/png')) for p in images]
        response=session.post(base+'/v1/images/edits',data={k:str(v) for k,v in payload.items()},files=files,timeout=1200)
    else:response=session.post(base+'/v1/images/generations',json=payload,timeout=1200)
    elapsed=time.perf_counter()-start
    if response.status_code!=200:
        (out/(name+'-error.json')).write_text(response.text)
        raise RuntimeError(f'{name} HTTP {response.status_code}: {response.text[:600]}')
    body=response.json();encoded=body['data'][0].pop('b64_json');raw=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(raw));image.load();array=np.asarray(image)
    expected=tuple(map(int,size.split('x')))
    assert image.size==expected and array[...,:3].std()>1
    path=out/(name+'.png');path.write_bytes(raw);arrays[name]=array
    row={'case':name,'prompt':prompt,'size':size,'seed':seed,'generator_device':'cpu','http_status':200,
         'condition_image_count':len(images or []),'request_seconds':elapsed,'image_mode':image.mode,
         'image_sha256':hashlib.sha256(raw).hexdigest(),'pixel_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
         'pixel_standard_deviation':float(array[...,:3].std()),'response_metadata':body}
    if name=='text512':
        ref=ROOT/'results/scnet/runtime-20261026T124357Z/smoke512/text-to-image-512.png'
        reference=np.asarray(Image.open(ref));assert reference.shape==array.shape
        delta=np.abs(reference.astype(float)-array.astype(float))
        row['same_BW_diffusers_reference']={'pixels_identical':bool(np.array_equal(reference,array)),
                                          'mean_absolute_pixel_difference':float(delta.mean()),'max_pixel_difference':float(delta.max())}
    if name=='text512-repeat':
        row['same_seed_pixels_identical']=bool(np.array_equal(array,arrays['text512']))
        assert row['same_seed_pixels_identical'],'Fixed seed did not reproduce the same pixels'
    if name=='text512-otherseed':
        row['different_seed_pixels_differ']=not bool(np.array_equal(array,arrays['text512']))
        assert row['different_seed_pixels_differ'],'Different seeds unexpectedly returned identical pixels'
    rows.append(row);(out/'summary.json').write_text(json.dumps({'cases':rows,'official_accuracy_pass':None},indent=2))
    print(json.dumps(row),flush=True)
    return path


prompt='A red ceramic teapot on a wooden table beside a blue cup, soft daylight, clean studio photograph.'
teapot=call('text512',prompt,'512x512')
call('text512-repeat',prompt,'512x512')
call('text512-otherseed',prompt,'512x512',seed=43)
bird=call('bird512','A small yellow bird perched on a branch against a pale green background, detailed studio photograph.','512x512')
call('text1024','A clean white poster with bold dark blue Chinese text "国芯国模" and a simple red geometric circle, careful typography.','1024x1024')
call('text2048',prompt,'2048x2048')
call('edit-one','Change only the red teapot to green, keeping the blue cup and the wooden table.','512x512',[teapot])
call('edit-two','Place the teapot from the first reference and the yellow bird from the second reference together on a wooden table.','512x512',[teapot,bird])
print('All eight SGLang generic Diffusers image API cases completed',flush=True)
