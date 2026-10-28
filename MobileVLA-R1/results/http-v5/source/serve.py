"""RGB MobileVLA frontend with the original checkpoint and vLLM language backend."""
import argparse,ast,base64,dataclasses,hashlib,io,json,logging,sys,threading,time,types
from contextlib import asynccontextmanager
from enum import Enum,auto
from typing import List
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer,SiglipVisionModel,SiglipImageProcessor
from fastapi import FastAPI,HTTPException
from vllm import LLM,SamplingParams
import uvicorn

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/code-reference'
def upstream(relative,names,namespace):
    raw=(SOURCE/relative).read_bytes()
    manifest=json.loads((ROOT/'sources/reference-files.json').read_text())
    expected=next(r['sha256'] for r in manifest['files'] if r['path']==relative)
    assert hashlib.sha256(raw).hexdigest()==expected
    tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names
           or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/relative),'exec'),namespace)

class Pipeline:
    def __init__(self):
        w=ROOT/'weights/mobilevla-r1-rl'
        assert json.loads((ROOT/'results/archive-check.json').read_text())['passed']
        self.tokenizer=AutoTokenizer.from_pretrained(w/'llm',local_files_only=True)
        self.processor=SiglipImageProcessor.from_pretrained(w/'vision_tower',local_files_only=True)
        # TF5 exposes crop_size=None; the original helper expected the attribute to be absent.
        if getattr(self.processor,'crop_size',None) is None:self.processor.crop_size=dict(self.processor.size)
        self.vision,info=SiglipVisionModel.from_pretrained(w/'vision_tower',local_files_only=True,
            dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
        self.vision.eval()
        namespace={'torch':torch,'nn':nn}
        upstream('llava/model/multimodal_projector/base_projector.py',{'DownSampleBlock'},namespace)
        self.projector=nn.Module();self.projector.layers=nn.Sequential(namespace['DownSampleBlock'](),
            nn.LayerNorm(4608),nn.Linear(4608,4096),nn.GELU(),nn.Linear(4096,4096))
        self.projector.load_state_dict(load_file(w/'mm_projector/model.safetensors'),strict=True)
        self.projector.to(device='cuda',dtype=torch.bfloat16).eval()
        index=json.loads((w/'llm/model.safetensors.index.json').read_text())
        shard=index['weight_map']['model.embed_tokens.weight']
        with safe_open(w/'llm'/shard,framework='pt',device='cpu') as tensors:
            embeddings=tensors.get_tensor('model.embed_tokens.weight')
        self.embeddings=nn.Embedding.from_pretrained(embeddings,freeze=True).to('cuda')
        module=types.ModuleType('mobilevla_serve_conversation');sys.modules[module.__name__]=module
        module.__dict__.update(dataclasses=dataclasses,Enum=Enum,auto=auto,List=List,logger=logging.getLogger('mobilevla'))
        upstream('llava/conversation.py',{'SeparatorStyle','Conversation','llama_3_chat'},module.__dict__)
        self.conversation=module.llama_3_chat
        namespace={'torch':torch,'IMAGE_TOKEN_INDEX':-200,'Image':Image}
        upstream('llava/mm_utils.py',{'tokenizer_image_token','process_image'},namespace)
        self.tokenize_image=namespace['tokenizer_image_token']
        self.process_image=namespace['process_image']
        namespace={}
        upstream('mobilevla/data.py',{'observation_question'},namespace)
        self.observation_question=namespace['observation_question']
        self.engine=LLM(model=str(w/'llm'),dtype='bfloat16',trust_remote_code=False,
            enable_prompt_embeds=True,enforce_eager=True,gpu_memory_utilization=.5,
            max_model_len=4096,max_num_seqs=1,enable_prefix_caching=False)
        self.lock=threading.Lock()

    @torch.inference_mode()
    def generate(self,images,question,max_tokens,temperature,seed):
        cfg=json.loads((ROOT/'weights/mobilevla-r1-rl/config.json').read_text())
        assert cfg['mm_vision_select_layer']==-2 and cfg['mm_vision_select_feature']=='cls_patch'
        conv=self.conversation.copy()
        conv.append_message(conv.roles[0],self.observation_question(question,['rgb']*len(images)))
        conv.append_message(conv.roles[1],None)
        ids=self.tokenize_image(conv.get_prompt(),self.tokenizer,return_tensors='pt').to('cuda')
        locations=torch.where(ids==-200)[0].tolist()
        assert len(locations)==len(images)
        args=types.SimpleNamespace(image_processor=self.processor,image_aspect_ratio=cfg['image_aspect_ratio'])
        pixels=torch.stack([self.process_image(image,args,None) for image in images]).to('cuda',dtype=torch.bfloat16)
        features=self.projector.layers(self.vision(pixels,output_hidden_states=True).hidden_states[-2])
        assert features.shape==(len(images),196,4096) and torch.isfinite(features).all()
        chunks=[];left=0
        for i,at in enumerate(locations):
            chunks.extend([self.embeddings(ids[left:at]),features[i]]);left=at+1
        chunks.append(self.embeddings(ids[left:]));embeds=torch.cat(chunks,dim=0)
        if embeds.shape[0]+max_tokens>4096:raise ValueError('Input plus output exceeds 4096 tokens')
        result=self.engine.generate([{'prompt_embeds':embeds.cpu()}],
            SamplingParams(temperature=temperature,top_p=.9 if temperature else 1,
                max_tokens=max_tokens,seed=seed,stop_token_ids=[128001,128009]),use_tqdm=False)[0]
        answer=result.outputs[0]
        return {'text':answer.text,'output_ids':list(answer.token_ids),'finish_reason':answer.finish_reason,
            'input_tokens':embeds.shape[0],'output_tokens':len(answer.token_ids),'images':len(images)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=30006);a=p.parse_args()
    pipeline=Pipeline()
    @asynccontextmanager
    async def lifespan(app):
        yield
        pipeline.engine.llm_engine.engine_core.shutdown(timeout=30)
    app=FastAPI(lifespan=lifespan)
    @app.get('/health')
    def health():return {'ready':True,'model':'mobilevla-r1','backend':'vllm','modalities':['rgb','text'],'max_images':8,'max_context':4096}
    @app.post('/generate')
    def generate(body:dict):
        try:
            images=body.get('images',[]);question=body.get('prompt','');limit=body.get('max_tokens',128)
            if not isinstance(images,list) or not 1<=len(images)<=8:raise ValueError('Provide 1..8 base64 RGB images')
            if not isinstance(question,str) or not question.strip() or '<image>' in question:raise ValueError('Provide nonempty prompt without image placeholder')
            if not isinstance(limit,int) or not 1<=limit<=512:raise ValueError('max_tokens must be 1..512')
            if any(k in body for k in ['depth','point_cloud']):raise ValueError('Only released RGB checkpoint path is implemented')
            decoded=[]
            for data in images:
                if not isinstance(data,str) or len(data)>6*1024*1024:raise ValueError('Invalid image size')
                raw=base64.b64decode(data.split(',',1)[-1],validate=True)
                image=Image.open(io.BytesIO(raw));image.load();decoded.append(image.convert('RGB'))
            temperature=float(body.get('temperature',0))
            if not 0<=temperature<=2:raise ValueError('temperature must be 0..2')
            start=time.perf_counter()
            with pipeline.lock:result=pipeline.generate(decoded,question,limit,temperature,int(body.get('seed',42)))
            return dict(result,seconds=time.perf_counter()-start)
        except (ValueError,TypeError) as e:raise HTTPException(status_code=400,detail=str(e))
    try:
        uvicorn.run(app,host='127.0.0.1',port=a.port)
    finally:
        pipeline.engine.llm_engine.engine_core.shutdown(timeout=30)

if __name__=='__main__':main()
