"""Run the publisher's unmodified backend on text, image and fused inputs."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from embedding_backends.common import ensure_transformers_special_tokens_compat,ensure_tokenizers_thread_safety
from embedding_backends.trident_qwen3vl import TridentQwen3VLBackend

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=json.loads((ROOT/'sources/upstream-source.json').read_text())
    for item in source['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    vendor=json.loads((ROOT/'sources/qwen-vl-utils-source.json').read_text())
    for item in vendor['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    ensure_transformers_special_tokens_compat();ensure_tokenizers_thread_safety()
    start=time.perf_counter()
    backend=TridentQwen3VLBackend({'model_type':'trident_qwen3vl','checkpoint':os.environ['MODEL_PATH'],
                                 'dtype':'bfloat16','role':'query'},device='cuda:0')
    backend.load();initialization=time.perf_counter()-start
    chart=str(ROOT/'sources/upstream/figs/example_chart.png')
    queries=[{'text':'How many stores did Saint Laurent operate in Western Europe in 2020?','images':[]}]
    docs=[{'text':'A bar chart showing the number of Saint Laurent stores by region and year.','images':[]},
          {'text':None,'images':[chart]},
          {'text':'Number of stores by region.','images':[chart]},
          {'text':'Tokyo is the capital of Japan.','images':[]},
          {'text':'Compare these two views of the same chart.','images':[chart,chart]}]
    vectors={};prompts={};tokens={};prompt_token_ids={};image_grids={}
    for role,items in [('query',queries),('doc',docs)]:
        backend.args_dict['role']=role
        instruction=backend.default_instruction(role)
        inputs=[{**x,'instruction':instruction} for x in items]
        prompts[role]=[backend.render_prompt(x.get('text'),x['images'],instruction,role) for x in inputs]
        assert not any(x.startswith('<render_prompt') for x in prompts[role])
        prepared=backend.build_inputs_cpu(inputs)
        prompt_token_ids[role]=[row[mask.bool()].tolist() for row,mask in zip(prepared['input_ids'],prepared['attention_mask'])]
        image_grids[role]=prepared['image_grid_thw'].tolist() if 'image_grid_thw' in prepared else []
        output,info=backend.process(inputs,role=role)
        assert output.shape==(len(items),2048) and np.isfinite(output).all()
        assert np.allclose(np.linalg.norm(output,axis=1),1,atol=1e-5)
        vectors[role]=output;tokens[role]=info
    out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
    scores=vectors['query']@vectors['doc'].T
    result={'backend':'publisher TridentQwen3VLBackend','dtype':'bfloat16','dimension':2048,
            'queries':queries,'documents':docs,'prompts':prompts,'token_counts':tokens,
            'prompt_token_ids':prompt_token_ids,'image_grid_thw':image_grids,
            'query_vectors':vectors['query'].tolist(),'document_vectors':vectors['doc'].tolist(),
            'scores':scores.tolist(),'ranking':scores[0].argsort()[::-1].tolist(),
            'initialization_seconds':initialization,'official_accuracy_pass':None,
            'validation_scope':'Publisher demonstration and distractor only; not a retrieval benchmark score.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in {'query_vectors','document_vectors','prompts'}},indent=2),flush=True)


if __name__=='__main__':main()
