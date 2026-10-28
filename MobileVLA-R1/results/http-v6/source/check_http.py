"""Synthetic HTTP functional checks; these are not navigation accuracy tests."""
import argparse,base64,io,json,time,urllib.request,urllib.error
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:30006')
    p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    def request(name,body=None,status=200):
        req=urllib.request.Request(a.url+('/health' if body is None else '/generate'),
            data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=180) as r:code=r.status;result=json.load(r)
        except urllib.error.HTTPError as e:code=e.code;result=json.loads(e.read())
        row={'name':name,'status':code,'expected_status':status,'response':result,'seconds':time.perf_counter()-start}
        rows.append(row);(out/'requests.json').write_text(json.dumps(rows,indent=2)+'\n')
        assert code==status,row
        return result
    def image(color):
        b=io.BytesIO();Image.new('RGB',(384,384),color).save(b,format='PNG')
        return base64.b64encode(b.getvalue()).decode()
    red=image((230,30,30));blue=image((30,30,230))
    body={'images':[red],'prompt':'Describe the visible colors and shapes.','max_tokens':64,'temperature':0,'seed':42}
    request('health')
    first=request('red',body);repeat=request('red-repeat',body)
    assert first['output_ids']==repeat['output_ids'] and first['output_tokens']>0
    alternate=request('blue',dict(body,images=[blue]))
    reference=json.loads((Path(__file__).resolve().parents[1]/'results/reference-probe-v5/summary.json').read_text())
    def strip(ids):return ids[:-1] if ids and ids[-1] in [128001,128009] else ids
    for value,label in [(first,'red'),(alternate,'blue')]:
        expected=next(x for x in reference['cases'] if x['case']==label)
        assert strip(value['output_ids'])==strip(expected['output_ids']),(label,value)
    for count in [2,8]:
        response=request(f'images-{count}',dict(body,images=([red,blue]*(count//2))))
        assert response['images']==count and response['output_tokens']>0
    for name,changes in [('no-images',{'images':[]}),('empty-prompt',{'prompt':''}),
                         ('invalid-base64',{'images':['%%%']}),('depth',{'depth':[]}),
                         ('too-many-images',{'images':[red]*9}),('invalid-budget',{'max_tokens':0})]:
        request(name,dict(body,**changes),status=400)
    (out/'summary.json').write_text(json.dumps({'passed':True,'checks':len(rows),
        'reference_token_parity':True,'synthetic_functional_only':True,'official_accuracy_tested':False},indent=2)+'\n')

if __name__=='__main__':main()
