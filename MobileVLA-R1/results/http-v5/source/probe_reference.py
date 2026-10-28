"""Model-local modular reference probe, not a serving or accuracy result."""
import ast,dataclasses,json,logging,sys,time,types,traceback
from enum import Enum,auto
from typing import List
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoTokenizer,LlamaForCausalLM,SiglipVisionModel,SiglipImageProcessor

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/code-reference'
OUT=ROOT/'results/reference-probe-v5';OUT.mkdir(parents=True,exist_ok=True)
def upstream_nodes(relative,names,namespace):
    tree=ast.parse((SOURCE/relative).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names
           or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes)==len(names),(relative,names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/relative),'exec'),namespace)

try:
    weights=ROOT/'weights/mobilevla-r1-rl'
    assert json.loads((ROOT/'results/archive-check.json').read_text())['passed']
    assert json.loads((ROOT/'results/extracted-rl.json').read_text())['archive_crc_verified']
    cfg=json.loads((weights/'config.json').read_text())
    assert cfg['mm_vision_select_feature']=='cls_patch' and cfg['mm_vision_select_layer']==-2 and not cfg['s2']
    ns={'torch':torch,'nn':nn}
    upstream_nodes('llava/model/multimodal_projector/base_projector.py',{'DownSampleBlock'},ns)
    projector=nn.Module();projector.layers=nn.Sequential(ns['DownSampleBlock'](),nn.LayerNorm(1152*4),
        nn.Linear(1152*4,4096),nn.GELU(),nn.Linear(4096,4096))
    info=projector.load_state_dict(load_file(weights/'mm_projector/model.safetensors'),strict=True)
    projector.to(device='cuda',dtype=torch.bfloat16).eval()
    vision,vinfo=SiglipVisionModel.from_pretrained(weights/'vision_tower',local_files_only=True,
        dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',output_loading_info=True)
    llm,linfo=LlamaForCausalLM.from_pretrained(weights/'llm',local_files_only=True,
        dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',output_loading_info=True)
    for name,loading in [('vision',vinfo),('llm',linfo)]:
        assert not loading.get('missing_keys') and not loading.get('unexpected_keys') and not loading.get('mismatched_keys'),(name,loading)
    vision.eval();llm.eval()
    tokenizer=AutoTokenizer.from_pretrained(weights/'llm',local_files_only=True)
    processor=SiglipImageProcessor.from_pretrained(weights/'vision_tower',local_files_only=True)
    if getattr(processor,'crop_size',None) is None:processor.crop_size=dict(processor.size)
    convmod=types.ModuleType('mobilevla_reference_conversation');sys.modules[convmod.__name__]=convmod
    convmod.__dict__.update(dataclasses=dataclasses,Enum=Enum,auto=auto,List=List,logger=logging.getLogger('reference'))
    upstream_nodes('llava/conversation.py',{'SeparatorStyle','Conversation','llama_3_chat'},convmod.__dict__)
    tokens_ns={'torch':torch,'IMAGE_TOKEN_INDEX':-200,'Image':Image}
    upstream_nodes('llava/mm_utils.py',{'tokenizer_image_token','process_image'},tokens_ns)
    observation_ns={};upstream_nodes('mobilevla/data.py',{'observation_question'},observation_ns)
    conv=convmod.llama_3_chat.copy()
    conv.append_message(conv.roles[0],observation_ns['observation_question']('Describe the visible colors and shapes.',['rgb']));conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt();ids=tokens_ns['tokenizer_image_token'](prompt,tokenizer,return_tensors='pt').to('cuda')
    assert int((ids==-200).sum())==1
    at=int(torch.where(ids==-200)[0][0]);rows=[]
    with torch.inference_mode():
        for label,color in [('red',(230,30,30)),('blue',(30,30,230))]:
            image=Image.new('RGB',(384,384),color)
            args=types.SimpleNamespace(image_processor=processor,image_aspect_ratio=cfg['image_aspect_ratio'])
            pixels=tokens_ns['process_image'](image,args,None).unsqueeze(0).to('cuda',dtype=torch.bfloat16)
            hidden=vision(pixels,output_hidden_states=True).hidden_states[-2]
            features=projector.layers(hidden)
            assert features.shape==(1,196,4096) and torch.isfinite(features).all()
            before=llm.get_input_embeddings()(ids[:at]);after=llm.get_input_embeddings()(ids[at+1:])
            embeds=torch.cat([before,features[0],after],dim=0).unsqueeze(0)
            cache=ROOT/'cache/reference-inputs-v5';cache.mkdir(parents=True,exist_ok=True)
            torch.save(embeds[0].cpu(),cache/(label+'.pt'))
            start=time.perf_counter()
            response=llm.generate(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],dtype=torch.long,device='cuda'),
                do_sample=False,max_new_tokens=64,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,output_scores=True)
            completion=response.sequences[0,-len(response.scores):].cpu().tolist()
            row={'case':label,'visual_shape':list(features.shape),'feature_mean':float(features.float().mean()),
                 'output_ids':completion,'text':tokenizer.decode(completion,skip_special_tokens=True),
                 'seconds':time.perf_counter()-start,'generation_steps':len(response.scores)}
            rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'reference_probe_only':True,'framework_serving_verified':False,
        'official_accuracy_tested':False,'loaded_without_missing_keys':True,'cases':rows},indent=2)+'\n')
except Exception as e:
    (OUT/'failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
