"""Whisper-large-v3 + the pinned official Seed-TTS English scoring function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import string
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/eval-deps'))
import jiwer
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from zhon.hanzi import punctuation

p=argparse.ArgumentParser();p.add_argument('--generation',required=True);a=p.parse_args()
folder=Path(a.generation)
raw_source=(ROOT/'sources/seedtts-reference/run_wer.py').read_text()
tree=ast.parse(raw_source)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_one')
namespace={'punctuation_all':punctuation+string.punctuation,'lang':'en','compute_measures':jiwer.compute_measures}
exec(compile(ast.Module(body=[function],type_ignores=[]),'official_seedtts_process_one','exec'),namespace)
score=namespace['process_one']
assert score('HELLO, world!','hello world')[2]==0
assert score('hello','hello world')[2]==0.5
assert json.loads((ROOT/'results/whisper-check/weights-check.json').read_text())['passed']
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(True)
model_path=ROOT/'evaluators/whisper-large-v3'
processor=WhisperProcessor.from_pretrained(model_path,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
      torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda:0').eval()
forced=processor.get_decoder_prompt_ids(language='english',task='transcribe')
manifest=json.loads((folder/'manifest.json').read_text())
with (folder/'generated.jsonl').open(encoding='utf-8') as stream:
    rows=[json.loads(line) for line in stream if line.strip()]
assert len(rows)==manifest['selected_rows'] and {r['id'] for r in rows}==set(manifest['ids'])
results=[];generation_keyword='forced_decoder_ids'
with (folder/'scored.jsonl').open('w') as log,torch.inference_mode():
    for row in rows:
        result={'id':row['id'],'target_text':row['target_text'],'passed':False}
        try:
            assert row['passed']
            path=folder/row['wav_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['response_sha256']
            audio,rate=sf.read(path)
            if rate!=16000:audio=scipy.signal.resample(audio,int(len(audio)*16000/rate))
            features=processor(audio,sampling_rate=16000,return_tensors='pt').input_features.to('cuda:0')
            started=time.perf_counter()
            if generation_keyword=='forced_decoder_ids':
                try:
                    predicted=model.generate(features,forced_decoder_ids=forced)
                except (ValueError,TypeError) as error:
                    if 'forced_decoder_ids' not in str(error):raise
                    generation_keyword='language_task'
                    predicted=model.generate(features,language='english',task='transcribe',return_timestamps=False)
            else:predicted=model.generate(features,language='english',task='transcribe',return_timestamps=False)
            transcript=processor.batch_decode(predicted,skip_special_tokens=True,clean_up_tokenization_spaces=True,
                clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True)[0]
            _,_,wer,subs,dele,inse=score(transcript,row['target_text'])
            result.update(transcript=transcript,wer=wer,substitution_rate=subs,deletion_rate=dele,insertion_rate=inse,
                          predicted_ids=predicted[0].cpu().tolist(),seconds=time.perf_counter()-started,passed=True)
        except Exception as error:result['error']=type(error).__name__+': '+str(error)
        results.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
        if len(results)%25==0 or not result['passed']:print(json.dumps({'scored':len(results),'last_passed':result['passed']}),flush=True)
errors=sum(not r['passed'] for r in results)
summary={'count':len(rows),'scored':len(results),'errors':errors,'full_dataset':manifest['full_dataset'],
         'mean_sample_wer_percent':float(np.mean([r['wer'] for r in results]))*100 if errors==0 else None,
         'wer_above_50_count':sum(r.get('wer',0)>.5 for r in results),
         'filter_high_error_samples':False,'asr_model':'openai/whisper-large-v3',
         'asr_revision':'06f233fe06e710322aca913c1bc4249a0d71fce1','asr_dtype':'float32',
         'generation_keyword':generation_keyword,'forced_decoder_ids_expected':forced,
         'legacy_bpe_cleanup_enabled':True,
         'score_source_revision':(ROOT/'sources/seedtts-reference/revision.txt').read_text().strip(),
         'normalizer_ast_sha256':hashlib.sha256(ast.dump(function).encode()).hexdigest(),
         'official_acceptance_claimed':False}
(folder/'wer-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
raise SystemExit(0 if errors==0 else 2)
