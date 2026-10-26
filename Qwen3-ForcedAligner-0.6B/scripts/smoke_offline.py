"""Run real audio/text alignment through native vLLM token classification."""
import json
import os
from pathlib import Path
import time
import librosa
import numpy as np
import soundfile as sf
import torch
from vllm import LLM
from upstream_processor import load_processor

ROOT=Path(__file__).resolve().parents[1]


def decode(output,fixture,config,processor,elapsed):
    logits=output.outputs.data
    assert isinstance(logits,torch.Tensor) and logits.ndim==2, type(logits)
    assert logits.shape[1]==config['thinker_config']['classify_num']
    assert torch.isfinite(logits).all(), 'Non-finite classifier output'
    ids=torch.tensor(output.prompt_token_ids,device=logits.device)
    assert len(ids)==logits.shape[0],(len(ids),tuple(logits.shape))
    timestamp_bins=logits.argmax(-1)[ids==config['timestamp_token_id']]
    assert len(timestamp_bins)==2*len(fixture['words'])
    raw_ms=(timestamp_bins*config['timestamp_segment_time']).cpu().numpy()
    spans=processor.parse_timestamp(fixture['words'],raw_ms)
    for span in spans:
        span['start_time']=round(span['start_time']/1000,3)
        span['end_time']=round(span['end_time']/1000,3)
    assert len(spans)==len(fixture['words']) and spans
    flat=[t for span in spans for t in [span['start_time'],span['end_time']]]
    assert all(0<=a<=b for a,b in zip(flat,flat[1:]))
    assert max(flat)<=fixture['audio_seconds']+0.16, (max(flat),fixture['audio_seconds'])
    record={'id':fixture['case']['id'],'language':fixture['case']['language'],
            'audio_seconds':fixture['audio_seconds'],'elapsed_seconds':elapsed,
            'classification_shape':list(logits.shape),'raw_timestamp_bins':timestamp_bins.cpu().tolist(),
            'items':spans,'format_monotonicity_bounds_passed':True}
    return record


def main():
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=proc.read_bytes().split(b'\0')
        except OSError:continue
        assert b'vllm.entrypoints.openai.api_server' not in args and b'sglang.launch_server' not in args, 'Stop other serving jobs first'
    out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
    model=Path(os.environ['MODEL_PATH']);config=json.loads((model/'config.json').read_text())
    processor=load_processor()
    cases=json.loads(Path(os.environ.get('ALIGNMENT_CASES',str(ROOT/'sources/functional-cases.json'))).read_text())
    payloads=[];fixtures=[]
    for case in cases:
        assert case['language'] in {'English','Chinese'}
        path=Path(os.environ.get('AUDIO_DIR',str(ROOT/'data/examples')))/case['audio_file']
        audio,sr=sf.read(path,dtype='float32')
        assert len(audio)>0 and sr>0 and np.isfinite(audio).all(), 'Invalid audio samples'
        if audio.ndim==2:audio=audio.mean(axis=1)
        if sr!=16000:audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
        words,prompt=processor.encode_timestamp(case['text'],case['language'])
        assert words, 'The provided text contains no alignable words or characters'
        payloads.append({'prompt':prompt,'multi_modal_data':{'audio':(np.asarray(audio,dtype=np.float32),16000)}})
        fixtures.append({'case':case,'words':words,'audio_seconds':len(audio)/16000})
    start=time.perf_counter()
    llm=LLM(model=str(model),runner='pooling',
            hf_overrides={'architectures':['Qwen3ASRForcedAlignerForTokenClassification']},
            dtype='bfloat16',gpu_memory_utilization=0.35,max_model_len=8192,max_num_seqs=8,
            enforce_eager=True,enable_prefix_caching=False,mm_processor_cache_gb=0)
    init_seconds=time.perf_counter()-start
    results=[]
    for fixture,payload in zip(fixtures,payloads):
        start=time.perf_counter()
        output=llm.encode([payload],pooling_task='token_classify')[0]
        elapsed=time.perf_counter()-start
        record=decode(output,fixture,config,processor,elapsed)
        results.append(record)
        (out/(record['id']+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    summary={'model':'Qwen/Qwen3-ForcedAligner-0.6B','framework':'vllm','dtype':'bfloat16',
             'runner':'pooling','pooling_task':'token_classify','enforce_eager':True,
             'initialization_seconds':init_seconds,'functional_cases':len(results),'cases':results,
             'validated_languages':['English','Chinese'],'official_accuracy_pass':None,
             'max_num_seqs':8,'precision_evaluation':'Timestamp format and bounds only; no AAS ground truth available.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if os.environ.get('BATCH_BENCH')=='1':
        bench=out/'bench';bench.mkdir(exist_ok=False)
        workloads=[];requests=[]
        reference={x['id']:x['items'] for x in results}
        for batch_size in [1,2,4,8]:
            count=max(64,4*batch_size)
            for round_id in [1,2]:
                llm.encode(payloads,pooling_task='token_classify',use_tqdm=False)
                total_wall=0.;total_audio=0.;max_delta=0.;exact=0
                for offset in range(0,count,batch_size):
                    indices=[i%len(payloads) for i in range(offset,offset+batch_size)]
                    start=time.perf_counter()
                    outputs=llm.encode([payloads[i] for i in indices],pooling_task='token_classify',use_tqdm=False)
                    elapsed=time.perf_counter()-start
                    assert len(outputs)==len(indices)
                    total_wall+=elapsed
                    for output,index in zip(outputs,indices):
                        row=decode(output,fixtures[index],config,processor,elapsed)
                        before=reference[row['id']];assert [x['text'] for x in before]==[x['text'] for x in row['items']]
                        delta=max(abs(a[k]-b[k]) for a,b in zip(before,row['items']) for k in ['start_time','end_time'])
                        max_delta=max(max_delta,delta);exact+=int(before==row['items'])
                        total_audio+=row['audio_seconds']
                        requests.append({'batch_size':batch_size,'round':round_id,'request':len(requests),
                                         'batch_elapsed_seconds':elapsed,'maximum_timestamp_delta_seconds':delta,**row})
                workload={'batch_size':batch_size,'round':round_id,'requests':count,'wall_seconds':total_wall,
                          'audio_seconds':total_audio,'audio_seconds_per_wall_second':total_audio/total_wall,
                          'same_as_serial':exact,'maximum_timestamp_delta_seconds':max_delta}
                workloads.append(workload);print(json.dumps(workload),flush=True)
        (bench/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in requests))
        report={'workloads':workloads,'counted_requests':len(requests),'round_reported':2,
                'timing_scope':'llm.encode only; audio I/O/resampling/text preparation excluded',
                'workload':'Two official demo WAVs with supplied English/Chinese text',
                'official_performance_protocol':False,'official_accuracy_pass':None}
        (bench/'summary.json').write_text(json.dumps(report,indent=2))



if __name__=='__main__':main()
