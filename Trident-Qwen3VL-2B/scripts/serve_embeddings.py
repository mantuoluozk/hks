"""Serial HTTP embedding bridge; text API plus explicit multimodal item extension."""
import argparse
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from PIL import Image
from embedding_backends.trident_qwen3vl import TridentQwen3VLBackend
from evaluate_chartqa import PromptFormatter
from native_runtime import load_model


class EmbeddingAdapter:
    def __init__(self,llm):
        self.llm=llm
        self.formatter=TridentQwen3VLBackend({'role':'query'},device='cpu')
        self.formatter.encoder=PromptFormatter()

    def encode(self,request):
        if not isinstance(request,dict):raise ValueError('Request must be a JSON object')
        role=request.get('role','query')
        if role not in {'query','doc'}:raise ValueError('role must be query or doc')
        if request.get('encoding_format','float')!='float':raise ValueError('Only float encoding is supported')
        if request.get('dimensions',2048)!=2048:raise ValueError('This checkpoint returns 2048 dimensions')
        items=request['input'];items=[items] if isinstance(items,(str,dict)) else items
        if not isinstance(items,list) or not 1<=len(items)<=8:raise ValueError('input must contain 1..8 items')
        payloads=[]
        for item in items:
            if isinstance(item,str):item={'text':item}
            if not isinstance(item,dict):raise ValueError('Each input is text or a text/images object')
            text=item.get('text');images=item.get('images',[])
            if text is not None and not isinstance(text,str):raise ValueError('text must be a string')
            if not isinstance(images,list) or len(images)>4 or any(not isinstance(p,str) for p in images):raise ValueError('images must contain at most four local image paths')
            if not text and not images:raise ValueError('Empty input')
            prompt=self.formatter.render_prompt(text,images,self.formatter.default_instruction(role),role)
            if prompt.startswith('<render_prompt'):raise ValueError(prompt)
            payload={'prompt':prompt}
            if images:payload['multi_modal_data']={'image':[Image.open(p).convert('RGB') for p in images]}
            payloads.append(payload)
        outputs=self.llm.embed(payloads,use_tqdm=False)
        vectors=np.asarray([x.outputs.embedding for x in outputs],dtype=np.float32)
        assert vectors.shape==(len(items),2048) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        tokens=sum(len(x.prompt_token_ids) for x in outputs)
        return {'object':'list','model':'Trident-Qwen3VL-2B','data':[{'object':'embedding','index':i,'embedding':v.tolist()} for i,v in enumerate(vectors)],'usage':{'prompt_tokens':tokens,'total_tokens':tokens}}


def make_server(adapter,host='127.0.0.1',port=8000):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            body=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path=='/health' else 404,{'ready':self.path=='/health'})

        def do_POST(self):
            if self.path!='/v1/embeddings':return self.reply(404,{'error':'Unknown endpoint'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1<<20:raise ValueError('JSON request must be between 1 byte and 1 MiB')
                value=adapter.encode(json.loads(self.rfile.read(length)))
            except (ValueError,KeyError,TypeError,OSError) as exc:return self.reply(400,{'error':str(exc)})
            except Exception as exc:return self.reply(500,{'error':type(exc).__name__+': '+str(exc)})
            self.reply(200,value)
    return HTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8000);args=parser.parse_args()
    adapter=EmbeddingAdapter(load_model());server=make_server(adapter,args.host,args.port)
    print('Embedding endpoint ready',server.server_address,flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
