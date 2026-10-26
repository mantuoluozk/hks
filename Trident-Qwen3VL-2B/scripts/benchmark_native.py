"""Repeated-fixture throughput; each timed cell covers 64 requests, not unique data."""
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image
from native_runtime import load_model


def main():
    ref=json.loads(Path(os.environ['REFERENCE_RESULTS']).read_text())
    llm=load_model()
    fixtures=[]
    for index,name in [(0,'text'),(1,'image'),(2,'fused')]:
        item=ref['documents'][index]
        payload={'prompt':ref['prompts']['doc'][index]}
        if item['images']:
            payload['multi_modal_data']={'image':[Image.open(p).convert('RGB') for p in item['images']]}
        fixtures.append((name,payload))
    out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
    cells=[]
    def fresh(payload,batch):
        records=[dict(payload) for _ in range(batch)]
        if 'multi_modal_data' in payload:
            for record in records:record['multi_modal_uuids']={'image':[uuid.uuid4().hex for _ in payload['multi_modal_data']['image']]}
        return records
    for round_id in [1,2]:
        for name,payload in fixtures:
            for batch in [1,2,4,8]:
                for _ in range(2):llm.embed(fresh(payload,batch),use_tqdm=False)
                elapsed=0;token_count=0
                for _ in range(64//batch):
                    inputs=fresh(payload,batch)
                    start=time.perf_counter()
                    outputs=llm.embed(inputs,use_tqdm=False)
                    elapsed+=time.perf_counter()-start
                    array=np.asarray([o.outputs.embedding for o in outputs],dtype=np.float32)
                    assert array.shape==(batch,2048) and np.isfinite(array).all()
                    assert np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-5)
                    token_count+=sum(len(o.prompt_token_ids) for o in outputs)
                row={'round':round_id,'fixture':name,'batch_size':batch,'requests':64,
                     'elapsed_seconds':elapsed,'requests_per_second':64/elapsed,
                     'input_tokens':token_count,'input_tokens_per_second':token_count/elapsed}
                cells.append(row);print(json.dumps(row),flush=True)
                (out/'summary.json').write_text(json.dumps({'cells':cells,'protocol':'Three fixed demonstration inputs repeated; sum of synchronous LLM.embed calls including preprocessing, model and result transfer. Excludes payload construction and output audit. Two warmups per cell; prefix and multimodal processor caches disabled; every image occurrence has a fresh UUID to prevent encoder-cache reuse.', 'cache_policy':'unique_image_uuid_per_request','official_accuracy_pass':None},indent=2))


if __name__=='__main__':main()
