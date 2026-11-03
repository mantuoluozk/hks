"""Pinned weights/audio/native sources and environment; never download or infer."""
import hashlib,importlib.metadata as metadata,json,os,shutil,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ['RUN_RESULTS_DIR']);OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()
model=Path(os.environ['MODEL_PATH']);entries=json.loads((ROOT/'sources/weight-files.json').read_text())['Data']['Files']
entries=[x for x in entries if x['Type']=='blob' and x['Path'] not in ['README.md','.gitattributes']];assert len(entries)==11
proof=[]
for x in entries:
    p=model/x['Path'];assert p.stat().st_size==x['Size'] and sha(p)==x['Sha256'],x['Path']
    proof.append({'path':x['Path'],'bytes':x['Size'],'sha256':x['Sha256'],'source_revision':x['Revision']});print('WEIGHT_OK',x['Path'],flush=True)
(OUT/'weights-check.json').write_text(json.dumps({'passed':True,'count':11,'files':proof,'excluded_non_runtime':['README.md','.gitattributes']},indent=2)+'\n')
old=Path(os.environ['OLD_ASR_ROOT']);audio=[]
for x in json.loads((ROOT/'sources/audio-sources.json').read_text()):
    p=ROOT/'data/examples'/x['file']
    if not p.exists():p=old/'data/examples'/x['file']
    assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];audio.append({**x,'path':str(p)});print('AUDIO_OK',x['file'],flush=True)
(OUT/'audio-check.json').write_text(json.dumps({'passed':True,'files':audio},indent=2)+'\n')
for name in ['config.json','generation_config.json','preprocessor_config.json','chat_template.json','tokenizer_config.json']:shutil.copyfile(model/name,ROOT/'sources'/('publisher-'+name))
import torch,sglang
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.multimodal.processors.qwen3_asr import Qwen3ASRMultimodalProcessor
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import Qwen3ASRAdapter
print('NATIVE_IMPORTS_OK',flush=True)
base=Path(sglang.__file__).parent;native={}
for rel,name in [('srt/models/qwen3_asr.py','model.py'),('srt/models/qwen3_omni_moe.py','audio_encoder.py'),('srt/multimodal/processors/qwen3_asr.py','processor.py'),('srt/entrypoints/openai/transcription_adapters/qwen3_asr.py','transcription_adapter.py'),('srt/entrypoints/openai/serving_transcription.py','serving_transcription.py'),('srt/configs/qwen3_asr.py','config.py')]:
    p=base/rel;dest=ROOT/'sources/native'/name
    if dest.exists():assert p.read_bytes()==dest.read_bytes(),rel
    else:shutil.copyfile(p,dest)
    native[rel]=sha(p)
packages={}
for n in ['torch','sglang','triton','transformers','tokenizers','sglang-das','sglang-hcu','sgl-kernel','aiter','librosa','soxr','soundfile','editdistance','openai','fastapi','python-multipart','numpy']:
    try:packages[n]=metadata.version(n)
    except metadata.PackageNotFoundError:pass
record={'hostname':socket.gethostname(),'python':sys.version,'packages':packages,'hip':torch.version.hip,
        'environment_source':os.environ.get('HKS_ENV_SOURCE'),'hks_root':os.environ['HKS_ROOT'],
        'gpus':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0),'memory':torch.cuda.mem_get_info(),
        'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        'shared_storage':dict(zip(['total','used','free'],shutil.disk_usage('/root/private_data'))),
        'region':'华中一区【A区】','image':'appstore.scnet.cn:5000/aihub/dcu/zack233/qwen-asr:v1',
        'dtk':Path('/opt/dtk/.dtk_version').read_text(),'native_source_sha256':native,'model_inference_verified':False}
(OUT/'environment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False),flush=True)
