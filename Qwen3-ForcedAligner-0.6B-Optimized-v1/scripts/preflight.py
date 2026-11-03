"""Inspect native pooling support and local files; never download weights."""
import importlib
import importlib.metadata as md
import json
import os
import hashlib
from pathlib import Path
import torch

out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
model=Path(os.environ['MODEL_PATH'])
module=importlib.import_module('vllm.model_executor.models.qwen3_asr_forced_aligner')
assert hasattr(module,'Qwen3ASRForcedAlignerForTokenClassification')
files=[{'path':p.relative_to(model).as_posix(),'bytes':p.stat().st_size}
       for p in model.rglob('*') if p.is_file()] if model.is_dir() else []
required=['config.json','tokenizer_config.json','preprocessor_config.json']
missing=[n for n in required if not (model/n).is_file()]
weights=[x for x in files if x['path'].endswith('.safetensors')]
result={'model':'Qwen/Qwen3-ForcedAligner-0.6B','model_path':str(model),
        'native_token_classification_import':True,'runner':'pooling','pooling_task':'token_classify',
        'packages':{n:md.version(n) for n in ['torch','vllm','transformers','openai','librosa']},
        'logical_device':os.environ.get('HIP_VISIBLE_DEVICES'),'hip':torch.version.hip,
        'files':files,'missing_configuration':missing,'weight_files':weights,
        'weights_available':bool(weights) and not missing,
        'inference_verified':False,'official_accuracy_pass':None,
        'stage':'ready_for_weight_provenance_check' if weights and not missing else 'awaiting_weights'}
result['devices']=[]
for i in range(torch.cuda.device_count()):
    d=torch.cuda.get_device_properties(i)
    result['devices'].append({'logical_device':i,'name':d.name,'gcn_arch':d.gcnArchName,
                              'memory':d.total_memory,'compute_units':d.multi_processor_count})
result['native_aligner_source_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
result['cpu_thread_environment']={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}
(out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,indent=2,ensure_ascii=False))
raise SystemExit(0 if result['weights_available'] else 2)
