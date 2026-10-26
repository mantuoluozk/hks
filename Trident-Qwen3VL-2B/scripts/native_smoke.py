"""Compare native vLLM LAST-token embeddings with the pinned publisher backend."""
import json
import os
from pathlib import Path
import time
import threading
import urllib.request
import urllib.error
import numpy as np
from PIL import Image
from native_runtime import load_model

ROOT=Path(__file__).resolve().parents[1]


def main():
    ref=json.loads(Path(os.environ['REFERENCE_RESULTS']).read_text())
    start=time.perf_counter()
    llm=load_model()
    initialization=time.perf_counter()-start
    vectors={};details=[];native_ids={}
    for role,items,key in [('query',ref['queries'],'query_vectors'),('doc',ref['documents'],'document_vectors')]:
        payloads=[]
        for item,prompt in zip(items,ref['prompts'][role]):
            payload={'prompt':prompt}
            if item['images']:
                payload['multi_modal_data']={'image':[Image.open(x).convert('RGB') for x in item['images']]}
            payloads.append(payload)
        start=time.perf_counter();outputs=llm.embed(payloads,use_tqdm=False);elapsed=time.perf_counter()-start
        actual=np.asarray([x.outputs.embedding for x in outputs],dtype=np.float32)
        expected=np.asarray(ref[key],dtype=np.float32)
        assert actual.shape==expected.shape and actual.shape[1]==2048
        assert np.isfinite(actual).all() and np.allclose(np.linalg.norm(actual,axis=1),1,atol=1e-5)
        cosines=np.sum(actual*expected,axis=1)
        native_ids[role]=[x.prompt_token_ids for x in outputs]
        equal=[a==b for a,b in zip(native_ids[role],ref['prompt_token_ids'][role])]
        details.append({'role':role,'count':len(items),'elapsed_seconds':elapsed,
                        'cosine_to_publisher_backend':cosines.tolist(),
                        'maximum_absolute_embedding_difference':float(np.max(np.abs(actual-expected))),
                        'prompt_token_ids_equal':equal})
        vectors[role]=actual
    scores=vectors['query']@vectors['doc'].T
    result={'backend':'native vLLM pooling convert=embed','dtype':'bfloat16','dimension':2048,
            'initialization_seconds':initialization,'comparisons':details,
            'query_vectors':vectors['query'].tolist(),'document_vectors':vectors['doc'].tolist(),
            'scores':scores.tolist(),'ranking':scores[0].argsort()[::-1].tolist(),
            'publisher_ranking':ref['ranking'],'native_prompt_token_ids':native_ids,
            'official_accuracy_pass':None,'comparison_scope':'Same BW device and inputs; not an official GPU precision baseline.'}
    out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {'query_vectors','document_vectors','native_prompt_token_ids'}},indent=2),flush=True)
    assert all(c>0.99 for x in details for c in x['cosine_to_publisher_backend']), 'Investigate implementation drift before further testing'
    from serve_embeddings import EmbeddingAdapter,make_server
    server=make_server(EmbeddingAdapter(llm),port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    api=[]
    try:
        for role,items in [('query',ref['queries']),('doc',ref['documents'])]:
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/embeddings',
                    data=json.dumps({'input':items,'role':role}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=120) as response:body=json.load(response)
            actual=np.asarray([r['embedding'] for r in body['data']],dtype=np.float32)
            cosines=np.sum(actual*vectors[role],axis=1)
            assert all(c>0.99 for c in cosines)
            api.append({'role':role,'count':len(items),'http_status':200,'cosine_to_direct_native':cosines.tolist(),'usage':body['usage']})
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/embeddings',data=b'{"input":"x","role":"invalid"}',headers={'Content-Type':'application/json'})
        try:urllib.request.urlopen(request);raise AssertionError('Invalid role accepted')
        except urllib.error.HTTPError as exc:assert exc.code==400
        (out/'api-summary.json').write_text(json.dumps({'cases':api,'invalid_role_status':400,'passed':True},indent=2))
    finally:server.shutdown();server.server_close();thread.join()
    if os.environ.get('RUN_BENCHMARK')=='1':
        import benchmark_native
        benchmark_native.load_model=lambda:llm
        os.environ['RUN_RESULTS_DIR']=str(out/'performance')
        benchmark_native.main()


if __name__=='__main__':main()
