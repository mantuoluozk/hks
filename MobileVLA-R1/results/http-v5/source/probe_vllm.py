"""Check vLLM's embedding-input path against saved modular reference inputs."""
import json,time,traceback,os
from pathlib import Path
import torch
from vllm import LLM,SamplingParams
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/vllm-probe-v2';OUT.mkdir(parents=True,exist_ok=True)
def main():
    try:
        reference=json.loads((ROOT/'results/reference-probe-v3/summary.json').read_text())
        assert reference['loaded_without_missing_keys']
        model=LLM(model=str(ROOT/'weights/mobilevla-r1-rl/llm'),dtype='bfloat16',
            trust_remote_code=False,enable_prompt_embeds=True,enforce_eager=True,
            gpu_memory_utilization=.5,max_model_len=4096,max_num_seqs=2,
            enable_prefix_caching=False)
        inputs=[]
        for label in ['red','blue']:
            tensor=torch.load(ROOT/'cache/reference-inputs'/(label+'.pt'),map_location='cpu',weights_only=True)
            assert tensor.ndim==2 and tensor.shape[1]==4096 and torch.isfinite(tensor).all()
            inputs.append({'prompt_embeds':tensor})
        started=time.perf_counter()
        outputs=model.generate(inputs,SamplingParams(temperature=0,max_tokens=64,stop_token_ids=[128009]),use_tqdm=False)
        cases=[]
        for original,output in zip(reference['cases'],outputs):
            result=output.outputs[0];expected=original['output_ids']
            actual=list(result.token_ids)
            def strip(ids):return ids[:-1] if ids and ids[-1] in [128009,128001] else ids
            cases.append({'case':original['case'],'text':result.text,'output_ids':actual,
                          'finish_reason':result.finish_reason,'token_ids_equal_to_reference':strip(actual)==strip(expected)})
        (OUT/'summary.json').write_text(json.dumps({'native_vllm_generation_executed':True,
            'end_to_end_http_serving_verified':False,'official_accuracy_tested':False,
            'cases':cases,'seconds':time.perf_counter()-started},indent=2)+'\n')
    except Exception as e:
        (OUT/'failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise

if __name__=='__main__':main()
