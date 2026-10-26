"""ChartQA Origin self-evaluation using publisher parsing and metrics, no merged Mix claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from embedding_backends.common import ensure_transformers_special_tokens_compat, ensure_tokenizers_thread_safety
from embedding_backends.trident_qwen3vl import TridentQwen3VLBackend
from mmemb.models.qwen3_vl_embedding import Qwen3VLEmbeddingEncoder

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources/upstream/src/eval'))
from embed_jsonl_unified_multigpu import parse_record


class PromptFormatter(Qwen3VLEmbeddingEncoder):
    """Use publisher formatting methods without allocating another GPU backbone."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        cfg=json.loads((Path(os.environ['MODEL_PATH'])/'mmemb_model.json').read_text())
        self.processor=AutoProcessor.from_pretrained(os.environ['MODEL_PATH'],padding_side='right')
        self.tokenizer=self.processor.tokenizer
        self.max_text_tokens=cfg['max_text_tokens'];self.image_root='';self.task_instructions={}
        self.query_instruction=cfg['prompt']['query_instruction'];self.doc_instruction=cfg['prompt']['doc_instruction']
        self.min_pixels=cfg['image']['min_pixels'];self.max_pixels=cfg['image']['max_pixels']
        self.fps=cfg['video']['fps'];self.max_frames=cfg['video']['max_frames'];self.total_pixels=cfg['video']['total_pixels']
        self.add_generation_prompt=cfg['add_generation_prompt']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['publisher','native'],required=True)
    args=parser.parse_args();ensure_transformers_special_tokens_compat();ensure_tokenizers_thread_safety()
    data=ROOT/'data/Trident-Eval/ChartQA'
    check=json.loads((ROOT/'results/dataset/chartqa-check.json').read_text());assert check['all_verified']
    out=Path(os.environ['RUN_RESULTS_DIR']);out.mkdir(parents=True,exist_ok=True)
    backend=TridentQwen3VLBackend({'checkpoint':os.environ['MODEL_PATH'],'dtype':'bfloat16','role':'query'},device='cuda:0')
    if args.backend=='publisher':backend.load()
    else:
        from native_runtime import load_model
        backend.encoder=PromptFormatter()
        ref=json.loads(Path(os.environ['REFERENCE_RESULTS']).read_text())
        for role,records in [('query',ref['queries']),('doc',ref['documents'])]:
            assert [backend.render_prompt(r['text'],r['images'],backend.default_instruction(role),role) for r in records]==ref['prompts'][role]
        llm=load_model()
    job_results=[]
    for role,name in [('query','queries'),('doc','origin_corpus')]:
        backend.args_dict['role']=role
        rows=[json.loads(line) for line in (data/(name+'.jsonl')).read_text().splitlines() if line.strip()]
        tasks=[parse_record(row,i,i+1,str(data),'id',role,backend.default_instruction(role),'remove',backend) for i,row in enumerate(rows)]
        assert all(t['_usable'] for t in tasks)
        directory=out/name;directory.mkdir(exist_ok=True)
        embeddings=[];counts=[];start=time.perf_counter()
        for offset in range(0,len(tasks),4):
            group=tasks[offset:offset+4]
            items=[{'text':t['_text'],'images':t['_image_paths'],'instruction':t['_instruction']} for t in group]
            if args.backend=='publisher':
                vectors,info=backend.process(items,role)
                nt=[x[0] for x in info]
            else:
                payloads=[]
                for item in items:
                    payload={'prompt':backend.render_prompt(item['text'],item['images'],item['instruction'],role)}
                    if item['images']:payload['multi_modal_data']={'image':[Image.open(p).convert('RGB') for p in item['images']]}
                    payloads.append(payload)
                outputs=llm.embed(payloads,use_tqdm=False)
                vectors=np.asarray([x.outputs.embedding for x in outputs],dtype=np.float32)
                nt=[len(x.prompt_token_ids) for x in outputs]
            assert vectors.shape==(len(group),2048) and np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
            embeddings.append(vectors)
            counts.extend({'id':t['_id'],'success':True,'input_tokens':n} for t,n in zip(group,nt))
            print(name,offset+len(group),'/',len(tasks),flush=True)
        np.save(directory/'embeddings.npy',np.concatenate(embeddings).astype(np.float16))
        (directory/'ids.txt').write_text(''.join(str(t['_id'])+'\n' for t in tasks))
        (directory/'token_counts.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in counts))
        job_results.append({'name':name,'records':len(tasks),'failed':0,'elapsed_seconds':time.perf_counter()-start})
    subprocess.run([sys.executable,str(ROOT/'sources/upstream/src/eval/eval_retrieval.py'),
                    '--queries_dir',str(out/'queries'),'--corpus_dir',str(out/'origin_corpus'),
                    '--qrels',str(data/'qrels.jsonl'),'--dataset_name','ChartQA','--device','cpu',
                    '--output_json',str(out/'origin-result.json'),'--dump_run',str(out/'rankings.jsonl')],check=True)
    (out/'summary.json').write_text(json.dumps({'backend':args.backend,'jobs':job_results,'embedding_save_dtype':'float16',
                  'scope':'Full ChartQA Origin subset only; not the six-dataset merged Mix benchmark.',
                  'dataset_revision':check['revision'],'official_accuracy_pass':None},indent=2))


if __name__=='__main__':main()
