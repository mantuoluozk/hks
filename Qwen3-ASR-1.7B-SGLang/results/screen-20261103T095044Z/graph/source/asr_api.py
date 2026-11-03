"""Native SGLang audio API; retain wire body and complete response after timing."""
import base64,hashlib,json,os,time
from pathlib import Path
import requests,soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:'+os.environ.get('PORT','30129')
MODEL=os.environ.get('SERVED_NAME','qwen3-asr-sglang')
def audio(path):
    path=Path(path);blob=path.read_bytes();info=sf.info(path)
    return {'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'audio_seconds':info.duration,
            'sample_rate':info.samplerate,'channels':info.channels,'data_url':'data:audio/'+path.suffix[1:]+';base64,'+base64.b64encode(blob).decode()}
def transcript(text):return (text or '').split('<asr_text>',1)[-1].strip()
def infer(item,mode='chat',seed=None,persist=True):
    session=requests.Session();session.trust_env=False
    body={'model':MODEL,'messages':[{'role':'user','content':[{'type':'audio_url','audio_url':{'url':item['data_url']}}]}],
          'temperature':0,'max_tokens':1024,'stream':mode=='sse'}
    if seed is not None:body['seed']=seed
    if mode=='sse':body['stream_options']={'include_usage':True}
    record={k:v for k,v in item.items() if k!='data_url' and not k.startswith('_')};record.update(mode=mode,error=None,success=False,decode={'temperature':0,'max_tokens':1024,'natural_eos':True,'language':None})
    if not persist:assert item.get('_request_ref'), 'Warm this exact fixture before timing'
    if mode=='transcription':record['decode']={'request_temperature':0,'native_adapter_effective_temperature':0.01,'native_adapter_max_new_tokens':256,'natural_eos':True,'language':None}
    response=None;started=time.perf_counter()
    try:
        if mode=='transcription':
            with Path(item['path']).open('rb') as f:response=session.post(BASE+'/v1/audio/transcriptions',data={'model':MODEL,'temperature':'0'},files={'file':(Path(item['path']).name,f,'audio/wav')},timeout=300)
        else:response=session.post(BASE+'/v1/chat/completions',json=body,stream=mode=='sse',timeout=300)
        response.raise_for_status()
        if mode=='sse':
            events=[];parts=[];done=False;usage=None;finish=None
            for line in response.iter_lines():
                if not line:continue
                events.append(line.decode())
                if line==b'data: [DONE]':done=True;break
                if line.startswith(b'data: '):
                    event=json.loads(line[6:])
                    if event.get('usage'):usage=event['usage']
                    if event.get('choices'):
                        c=event['choices'][0];parts.append(c.get('delta',{}).get('content') or '')
                        finish=c.get('finish_reason') or finish
            assert done,'SSE missing DONE'
            text=''.join(parts);raw={'raw_sse_lines':events,'done':done,'usage':usage};record['finish_reason']=finish
        else:
            raw=response.json()
            if mode=='transcription':text=raw.get('text','');record['finish_reason']=None
            else:text=raw['choices'][0]['message'].get('content') or '';record['finish_reason']=raw['choices'][0]['finish_reason']
        elapsed=time.perf_counter()-started
        record.update(response=raw,raw_output=text,text=transcript(text),elapsed_seconds=elapsed,rtf=elapsed/item['audio_seconds'],usage=raw.get('usage'),success=True)
    except Exception as e:
        elapsed=time.perf_counter()-started;record.update(error=repr(e),elapsed_seconds=elapsed,rtf=elapsed/item['audio_seconds'],text='')
        if response is not None:record['raw_error_body']=response.text
    if response is not None and persist:
        wire=response.request.body
        if isinstance(wire,str):wire=wire.encode()
        digest=hashlib.sha256(wire).hexdigest();p=ROOT/'data/raw-requests'/(digest+'.body');p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(wire)
        record['request']={'body_sha256':digest,'body_path':str(p),'content_type':response.request.headers.get('Content-Type'),'endpoint':response.request.path_url,'reference_or_language_supplied':False}
    elif not persist:record['request']=dict(item['_request_ref'])
    session.close();return record
